from tqdm import tqdm
import os, pickle, time
import sklearn as sk
import numpy as np
import sys
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./glove'))

def dumpingFiles(filePath, outFilename, files):
    dumpingPath = os.path.join(filePath, outFilename)
    print("Dumping at..", dumpingPath)
    with open(dumpingPath, 'wb') as outp:
        pickle.dump(files, outp, -1)
        
def loadingFiles(filePath, filename):
    loadingPath = os.path.join(filePath, filename)
    print("Loading at..", loadingPath)
    with open(loadingPath, 'rb') as f:
        p = pickle.load(f)
    return p

def get_conn(DB_connection_filePath='./DB_connection.txt'):
    import pymssql

    conn_dict = dict()
    with open(DB_connection_filePath, 'r') as f:
        for l in f.readlines():
            k, v = l.split('=')
            conn_dict[k] = v.split('\n')[0][1:-1]
            print("{}: {}".format(k, str(v.split('\n')[0])))
    conn = pymssql.connect(conn_dict['SERVER_IP'], conn_dict['USER'], conn_dict['PASSWD'], conn_dict['DB'])
    return conn

def query(conn, q, insert_items=None, df=False):
    import time
    cursor = conn.cursor()
    st_time = time.time()
    if insert_items is not None: cursor.executemany(q, insert_items)
    else: cursor.execute(q)
    if ('into' in q.lower()) or ('drop' in q.lower()) or ('create' in q.lower()):
        conn.commit()
        print("Done", time.time()-st_time)
        print("Executed")
        return None
    result = cursor.fetchall()
    if df:
        import pandas as pd
        result = pd.DataFrame(np.array(result), columns=[r[0] for r in cursor.description])
    conn.commit()
    print("Done", time.time()-st_time)
    #assert len(result)!=0, 'No RESULT!!!!'
    return result
    
def _making_condition_dict(conn, filePath='/data/hdd_5.5T/JIN/CDM', concept_dict_filename='CDM_concept_dict.pkl'):
    print("Making dict for [Searching from dict]..")
    q = str("select concept_id, concept_name from NHIS_NSC.dbo.CONCEPT "+
            "where DOMAIN_ID='Condition' and invalid_reason is null")
    dictionary = dict((concept_id, exp) for concept_id, exp in query(conn, q))
    dumpingFiles(filePath, concept_dict_filename, dictionary)
    
def _searching_from_dict(filePath, ref_dict_filename):
    print("Searching_from_dict")
    with open(os.path.join(filePath, ref_dict_filename), 'rb') as f: ref_dict = pickle.load(f)
    target = ''
    while(True):
        print("\nSearch item: ")
        item = input()
        target += '_'+item
        result_dict = dict( (key, val) for key, val in ref_dict.items() if str(val).lower().find(str(item).lower()) != -1 )
        for key, val in result_dict.items(): print(key, "\t:", val)
        print("\nnum of items.. {}\nStop searching? [y/n]".format(len(result_dict)))
        stop = input()
        if stop=='y': break
        else: ref_dict = result_dict
    if target.startswith('_'): target = target[1:]
    if target.endswith(' '): target = target[:-1]
    return result_dict, target.replace(' ','_')





#########################################################################################################

def Make_seqCohorts(conn, result_dict, q_age_condition, min_visit=2, pred_interval_unit='year', pred_interval=1, ratio=4, caliper=0.05, method='PSM', max_visit_clip=-1):
    
    def Making_init(conn, q_age_condition):
        def _get_age_conditions(q_age_condition):
            if q_age_condition:
                if not isinstance(q_age_condition, str):
                    print("\n\tEnter your age_condition query (for init_cohort)")
                    q_age_condition = input()
                if len(q_age_condition)!=0 and set(q_age_condition)!=' ': 
                    for q in q_age_condition.lower().split('and'):
                        if '<' in q: q_ceiling = q.strip()
                        else: q_floor = q.strip()
                else: 
                    print("Custom query is not good.. discarded")
                    q_ceiling, q_floor = '', ''
            else: q_ceiling, q_floor = '', ''
            return q_ceiling, q_floor
        
        print("\n@[1/3] Making_init")
        q_init_base = str("select co.person_id, co.visit_occurrence_id, co.condition_start_date, co.condition_concept_id, " 
                                 + " p.gender_concept_id, datepart(year, co.condition_start_date) - p.year_of_birth as visit_age " 
                             + " from condition_occurrence co " 
                             + " inner join (select person_id, gender_concept_id, year_of_birth, month_of_birth, day_of_birth from PERSON ) as p " 
                             + " on co.person_id = p.person_id ")

        q_init_condition = str("select person_id into #tmp_init_condition " 
                               + " from (select person_id, max(visit_age) as age " 
                                       + " from ( " + q_init_base + " )v group by person_id)vv {} ")

        q_init = str("select * into #tmp_init "
                     + " from ( " + q_init_base + " ) base "
                     + " where person_id in (select * from #tmp_init_condition) ")
        
        print("\nMaking.. 'Initial_cohort' (#tmp_init)")
        if q_age_condition: q_ceiling, q_floor = _get_age_conditions(q_age_condition)
        else: q_age_condition = ''
        
        if len(q_ceiling)>0: init_age_condition = ' where ' + q_ceiling
        else: init_age_condition = ''
        
        query(conn, q_init_condition.format(init_age_condition))
        query(conn, q_init)
        return q_age_condition, q_ceiling, q_floor

    
    def Making_seqTable(conn, q_age_condition, result_dict, min_visit, pred_interval_unit, pred_interval):

        def _make_treatment_group(conn, target_id_tuple, q_age_condition, min_visit, pred_interval_unit, pred_interval):    
            print("\nMaking.. #tmp_treat_target")
            q_target = str("select * into #tmp_treat_target " 
                                + " from (select *, row_number() over (partition by person_id order by condition_start_date) as target_rn " 
                                        + " from (select * from #tmp_treat where condition_concept_id in {})v )vv ")
            query(conn, q_target.format(target_id_tuple))

            print("\nMaking.. #tmp_treat_seq")
            q_treat_seq_base = str("select person_id, visit_occurrence_id, condition_start_date, condition_concept_id, gender_concept_id, visit_age, rn " 
                                  + " into #tmp_treat_seq_base " 
                                  + " from (select * from #tmp_treat base "
                                          + " inner join (select person_id as ref_pid, rn as ref_rn, condition_start_date as ref_date " 
                                                      + " from #tmp_treat_target where target_rn=1 {0})v "
                                          + " on base.person_id=v.ref_pid "
                                          + " where v.ref_rn >= {1} and base.rn < v.ref_rn and base.condition_start_date <= DATEADD({2}, {3}, v.ref_date) )vv ")


            #q_treat_seq = str("select * into #tmp_treat_seq from #tmp_treat_seq_base seq_base " 
            #                  + " join (select person_id, max(rn) as new_ref_rn from #tmp_treat_seq_base group by person_id)v "
            #                  + " on seq_base.person_id=v.person_id " 
            #                  + " where person_id in (select person_id "
            #                                      + " from (select person_id, max(rn) as new_ref_rn from #tmp_treat_seq_base group by person_id)vv " 
            #                                      + " where new_ref_rn >= {})")
            
            q_treat_seq = str("select * into #tmp_treat_seq " 
                          + " from (select seq_base.*, v.ref_rn from #tmp_treat_seq_base seq_base " 
                                  + " join (select person_id, max(rn) as ref_rn from #tmp_treat_seq_base group by person_id)v "
                                  + " on seq_base.person_id=v.person_id )vv " 
                          + " where ref_rn >= {} ")
            
            
            if len(q_age_condition)>0: 
                q_age_condition = ' and ' + q_age_condition.replace('age', 'visit_age')
            query(conn, q_treat_seq_base.format(q_age_condition, min_visit+1, pred_interval_unit, -pred_interval))
            query(conn, q_treat_seq.format(min_visit))
            print("\nDONE!!\n")


        def __get_demo(conn, group):
            q_demo = str("select person_id, max(gender_concept_id), max(visit_age), max(rn), cast((max(visit_age)-min(visit_age)) as float) / max(rn) "
                         + " from #tmp_{} group by person_id").format(group)
            group_pid = []
            group_demo = []
            for pid, gid, max_age, max_rn, avg_gap in query(conn, q_demo):
                group_pid.append(pid)
                if str(gid)=='8507': g=1.0
                else: g=2.0
                group_demo.append([g, float(max_age), float(max_rn), float(avg_gap)])
            return np.array(group_pid), np.array(group_demo)

        def __matching_cohorts(treat, control, ratio, caliper, method):
            import time
            from sklearn.preprocessing import StandardScaler

            st_time = time.time()
            print("\nMatching by.. {}".format(method))
            scaler = StandardScaler()
            scaler.fit(treat)
            treat_scaled = scaler.transform(treat)
            control_scaled = scaler.transform(control)

            assert_value = False
            if method=='PSM':
                from sklearn.linear_model import LogisticRegression
                t = np.append(np.ones([treat_scaled.shape[0]]).reshape([-1,1]), treat_scaled, axis=1)
                c = np.append(np.zeros([control_scaled.shape[0]]).reshape([-1,1]), control_scaled, axis=1)
                data = np.concatenate([t,c], axis=0)

                propensity = LogisticRegression(penalty='l2', class_weight='balanced', solver='sag', max_iter=10000, verbose=1, n_jobs=-1)
                propensity = propensity.fit(data[:,1:], data[:,0])
                score = propensity.predict_proba(data[:,1:])[:,1] # score for class 1 (T)
                t_score, c_score = np.split(score, [len(t)])

                #needs for tuning.. matrix_operation
                matched_indices = []
                for t_idx, t_s in enumerate(tqdm(t_score)):
                    distance = abs(t_s-c_score)
                    candidate = np.array([[c_idx, d] for c_idx, d in enumerate(distance) if d <= caliper])
                    if len(candidate)>0:
                        matched_indices.append(candidate[:,0][candidate.argsort(axis=0)[:,1][:ratio]])
                    else: matched_indices.append([])
                assert_value = True

            elif method=='KNN':
                from sklearn.neighbors import NearestNeighbors
                nbrs = NearestNeighbors(n_neighbors=ratio, algorithm='auto', p=2, n_jobs=-1).fit(control_scaled)
                matched_distances, matched_indices = nbrs.kneighbors(treat_scaled)
                assert_value = True

            assert assert_value == True, "Matching_method should be in ['PSM', 'KNN']"
            print("Done", time.time()-st_time)
            return matched_indices

        def _make_matched_control_group(conn, q_age_condition, min_visit, ratio=4, caliper=0.05, method='PSM'):
            from itertools import chain

            print("\nFetching.. treat_demo from #tmp_treat_seq")
            treat_pid, treat_demo = __get_demo(conn, 'treat_seq')
            print("\nFetching.. control_demo from #tmp_control")
            control_pid, control_demo = __get_demo(conn, 'control')
            matched_indices = __matching_cohorts(treat_demo, control_demo, ratio, caliper, method)
            matched_indices_unique = np.array(list(set(chain.from_iterable(matched_indices))), dtype=np.int) 
            assert len(matched_indices_unique)!=0, "No Matching Results.. by {}".format(method)
            matched_control_pid = control_pid[matched_indices_unique]

            query(conn, str("CREATE TABLE #tmp_control_matched_pid (person_id INT NOT NULL)"))
            query(conn, "INSERT INTO #tmp_control_matched_pid (person_id) VALUES (%d)", matched_control_pid.tolist())

            print("\nMaking.. #tmp_control_seq")
            q_control_seq_base = str("select * into #tmp_control_seq{} " 
                                + " from (select base.*, v.max_rn " 
                                        + " from (select person_id, visit_occurrence_id, condition_start_date, condition_concept_id, gender_concept_id, visit_age, rn " 
                                                + " from #tmp_control where person_id in (select distinct person_id from #tmp_control_matched_pid)) base " 
                                        + " join (select person_id, max(rn) as max_rn from #tmp_control where person_id in (select distinct person_id from #tmp_control_matched_pid) group by person_id) v " 
                                        + " on base.person_id = v.person_id) vv " 
                                + " where max_rn >= {}")
            query(conn, q_control_seq_base.format(' ', min_visit))
            #query(conn, q_control_seq_base.format('_base', min_visit))

            #q_control_seq = str("select * into #tmp_control_seq from #tmp_control_seq_base " 
            #                    + " where person_id in (select person_id from (select person_id, max(visit_age) as age " 
            #                                                                + " from #tmp_control_seq_base group by person_id)v " 
            #                                        + " {}) ")

            #if len(q_age_condition.strip())>0: q_age_condition = ' where ' + q_age_condition.strip()
            #query(conn, q_control_seq.format(q_age_condition))
            print("\nDONE!!\n")
            return treat_pid, matched_control_pid


        print("\n@[2/3] Making_seqTable")
        q_treat_pid = str("select distinct person_id into #tmp_treat_pid from #tmp_init" 
                        + " where condition_concept_id in (select concept_id from concept where DOMAIN_ID='Condition' and concept_id in {0} )")

        q_control_pid = str("select distinct person_id into #tmp_control_pid from #tmp_init" 
                        + " where person_id not in (select person_id from #tmp_treat_pid)")

        q_base = str("select * into #tmp_{0}_base " 
                     + " from (select person_id, visit_occurrence_id, condition_start_date, row_number() over (partition by person_id order by condition_start_date) as rn " 
                             + " from (select person_id, visit_occurrence_id, min(condition_start_date) condition_start_date from #tmp_init " 
                                     + " where person_id in (select person_id from #tmp_{0}_pid) " 
                                     + " group by person_id, visit_occurrence_id)b )vv ")

        q_visit = str("select * into #tmp_{0} " 
                      + " from (select visit.* "
                              + " from (select base.person_id, base.visit_occurrence_id, base.condition_start_date, base.rn, init.condition_concept_id, " 
                                                + " init.gender_concept_id, init.visit_age" 
                                      + " from (select * from #tmp_{0}_base) base "
                                      + " inner join (select * from #tmp_init where person_id in (select person_id from #tmp_{0}_pid)) init " 
                                      + " on base.person_id = init.person_id and base.visit_occurrence_id = init.visit_occurrence_id) visit ) v ")

        target_id_tuple = tuple(result_dict.keys())
        if len(target_id_tuple)==1: target_id_tuple = '(' + str(target_id_tuple[0]) + ')'

        for group in ['treat', 'control']:
            print("\nMaking.. #tmp_{}".format(group))
            if group=='treat': q_pid = q_treat_pid
            else: q_pid = q_control_pid
            query(conn, q_pid.format(target_id_tuple))
            query(conn, q_base.format(group))
            query(conn, q_visit.format(group))

            print("\nMaking_{}_seqTable..".format(group))
            if group=='treat': _make_treatment_group(conn, target_id_tuple, q_age_condition, min_visit, pred_interval_unit, pred_interval)
            else: treat_pid, matched_control_pid = _make_matched_control_group(conn, q_age_condition, min_visit, ratio, caliper, method)

        return treat_pid, matched_control_pid
    
    
    def Making_truncated_seqTable(conn, max_visit_clip):
        print("\n@[3/3] Making_truncated_seqTable")
        print("\nGetting max_time_step..")
        q_max_rn = str("select 1, max(rn) from #tmp_{}_seq UNION ALL select 0, max(rn) from #tmp_{}_seq")
        t_maxrn, c_maxrn = query(conn, q_max_rn.format('treat', 'control'))
        if max_visit_clip != -1: max_time_step = min(t_maxrn[1], c_maxrn[1], max_visit_clip)
        else: max_time_step = min(t_maxrn[1], c_maxrn[1])
        print("\tMAX_TIME_STEP: {}".format(max_time_step))

        for group, opt in [('treat', 'ref_rn'),('control', 'max_rn')]:
            print("\nMaking_truncated_{}_seqTable..".format(group))
            q_truncated = str("select * into {0}_seq "
                            + "from (select person_id, rn, condition_concept_id, gender_concept_id, visit_age " 
                                     + " from #tmp_{0}_seq where {1}<={2} UNION ALL " 
                                     + " select * from (select person_id, rn-{1}+{2} as new_rn, condition_concept_id, gender_concept_id, visit_age " 
                                                         + " from #tmp_{0}_seq where {1}>{2})v " 
                                     + " where new_rn>0) vv")
            query(conn, q_truncated.format(group, opt, max_time_step))
        
        return max_time_step
    
    
    ## MAIN..
    print("\n[Phase 1] Making cohorts\n")
    
    # Initial_cohort
    q_age_condition, q_ceiling, q_floor = Making_init(conn, q_age_condition)
    
    # T/C_cohort
    treat_pid, matched_control_pid = Making_seqTable(conn, q_age_condition, result_dict, min_visit, pred_interval_unit, pred_interval)
    
    # Truncated_cohorts
    max_time_step = Making_truncated_seqTable(conn, max_visit_clip)
    
    #info function.. 
    
    print("\nALL DONE@@\n")
    
    return max_time_step, treat_pid, matched_control_pid

    
def _get_code_dicts(conn, filePath):    
    import collections
    from itertools import chain
    
    st_time=time.time()
    q_flatCode = str("select condition_concept_id from {}_seq UNION ALL select condition_concept_id from {}_seq")
    rawSeqs_flatted = list(chain.from_iterable(query(conn, q_flatCode.format('treat', 'control'))))
    
    code_cnt = collections.Counter(rawSeqs_flatted).most_common()
    #with open(os.path.join(filePath, 'CDM_concept_dict.pkl'), 'rb') as f: ref_dict = pickle.load(f) 
    #print("Top 10 common codes and cnt:")
    #for i in range(10): print("\t", code_cnt[i][0], "\t", code_cnt[i][1], 
    #                          "\t", ref_dict[code_cnt[i][0]])
    code_to_id = {code: i for i, (code, cnt) in enumerate(code_cnt)}
    id_to_code = {val:key for key, val in code_to_id.items()}
    print("\n\tTakes..", time.time()-st_time)
    return rawSeqs_flatted, code_cnt, code_to_id, id_to_code

def _get_pid2visit_multiHot(conn, max_time_step, code_to_id):
    
    def _convert_to_sparseMultiHot(conn, group, code_to_id, max_time_step, feature_size):
        import collections
        from scipy.sparse import csr_matrix
        
        pid2rn = collections.defaultdict(list)
        pid2cid = collections.defaultdict(list)

        q_code = str("select person_id, rn, condition_concept_id, gender_concept_id, visit_age " 
                     + " from {}_seq order by person_id, rn")
        for pid, rn, cid, gid, age in query(conn, q_code.format(group)):
            pid2rn[pid].append(rn-1)
            pid2cid[pid].append(code_to_id[cid])

        pid2demo = collections.defaultdict(list)
        q_demo = str("select distinct person_id, rn, gender_concept_id, visit_age " 
                     + " from {}_seq order by person_id, rn")
        for pid, rn, gid, age in query(conn, q_demo.format(group)):
            if str(gid)=='8507': visits_demo = [1.0, 0.0, age]
            else: visits_demo = [0.0, 1.0, age]
            pid2demo[pid].append(visits_demo)

        pid2visit_multiHot = dict()
        pid2visit_demo_truncated = dict()
        for pid in pid2rn.keys():
            pid2visit_multiHot[pid] = [csr_matrix((np.ones((len(pid2cid[pid])), np.float32), (pid2rn[pid], pid2cid[pid])), 
                                                          shape=(max_time_step, feature_size)), len(set(pid2rn[pid]))]

            row_demo_idx = np.array([[idx,idx,idx] for idx in range(len(set(pid2rn[pid])))]).reshape([-1])
            col_demo_idx = np.array([[0,1,2] for idx in range(len(set(pid2rn[pid])))]).reshape([-1])    
            pid2visit_demo_truncated[pid] = csr_matrix(( np.array(pid2demo[pid], np.float32).reshape([-1]), 
                                                                 (row_demo_idx, col_demo_idx) ), 
                                                          shape=(max_time_step, 3)) #row_idx
        return pid2visit_multiHot, pid2visit_demo_truncated
    
    feature_size = len(code_to_id)
    print("Get_pid2visit_multiHot\n    max_time_step: {}\n    feature_size: {}".format(max_time_step, feature_size))
    print("\tcontrol_group")
    control_pid2visit_multiHot, control_pid2visit_demo_truncated = _convert_to_sparseMultiHot(conn, 'control', code_to_id, max_time_step, feature_size)
    print("\ttreatment_group")
    treatment_pid2visit_multiHot, treatment_pid2visit_demo_truncated = _convert_to_sparseMultiHot(conn, 'treat', code_to_id, max_time_step, feature_size)
    return control_pid2visit_multiHot, control_pid2visit_demo_truncated, treatment_pid2visit_multiHot, treatment_pid2visit_demo_truncated

def _split_data(group, pid2visit_multiHot, pid2visit_demo_truncated, n_samples=None):
    pid_list = list(pid2visit_multiHot.keys()) #[key for key in pid2visit_multiHot.keys()]
    if n_samples is not None: pid_list = sk.utils.resample(pid_list, n_samples=n_samples, replace=False)
    pid_list_split = np.split(pid_list, [int(len(pid_list)*0.8), int(len(pid_list)*0.9)])
    
    tmp_data_list = []
    for pid_list in pid_list_split:
        #print(len(pid_list))
        inputs = []
        seq_lens = []
        demo = []
        for pid in pid_list:
            inputs.append(pid2visit_multiHot[pid][0])
            seq_lens.append(pid2visit_multiHot[pid][1])
            demo.append(pid2visit_demo_truncated[pid])
        if group == 'treat': labels = [1]*len(pid_list)
        else: labels = [0]*len(pid_list)
        tmp_data_list.append([inputs, labels, seq_lens, demo])
    return pid_list_split, tmp_data_list

class __DataSet(object):
    def __init__(self, pids, data):
        self._num_examples = len(pids)
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._pids = pids.tolist()
        self._inputs, self._labels, self._seq_lens, self._demo = data
    
    def _shuffle(self, pids, inputs, labels, seq_lens, demo):
        return sk.utils.shuffle(pids, inputs, labels, seq_lens, demo)
    
    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        if end<=self._num_examples:
            return self._pids[start:end], self._inputs[start:end], self._labels[start:end], self._seq_lens[start:end], self._demo[start:end]
        
        else:
            self._epochs_completed += 1
            num_of_short = batch_size-(self._num_examples-start)
            num_of_extra_batch = num_of_short // self._num_examples
            num_of_extra_example = num_of_short % self._num_examples
            self._epochs_completed += num_of_extra_batch
            self._index_in_epoch = num_of_extra_example
            
            tmp_pids=self._pids[start:]; tmp_inputs=self._inputs[start:]; tmp_labels=self._labels[start:]; tmp_seq_lens=self._seq_lens[start:]; tmp_demo=self._demo[start:]       
            self._pids, self._inputs, self._labels, self._seq_lens, self._demo = self._shuffle(self._pids, self._inputs, self._labels, self._seq_lens, self._demo)
            batch_pids = tmp_pids+self._pids*num_of_extra_batch+self._pids[0:num_of_extra_example]
            batch_inputs = tmp_inputs+self._inputs*num_of_extra_batch+self._inputs[0:num_of_extra_example]
            batch_labels = tmp_labels+self._labels*num_of_extra_batch+self._labels[0:num_of_extra_example]
            batch_seq_lens = tmp_seq_lens+self._seq_lens*num_of_extra_batch+self._seq_lens[0:num_of_extra_example] 
            batch_demo = tmp_demo+self._demo*num_of_extra_batch+self._demo[0:num_of_extra_example] 
            return batch_pids, batch_inputs, batch_labels, batch_seq_lens, batch_demo
    
class __concat_dataSets():
    def __init__(self, c_dataSets, t_dataSets):
        self._c_dataSets = c_dataSets
        self._t_dataSets = t_dataSets
        
    @property
    def pids(self):
        return np.array(self._c_dataSets._pids+self._t_dataSets._pids)
        
    @property
    def inputs(self):
        return np.array([sprs_mat.toarray() for sprs_mat in self._c_dataSets._inputs+self._c_dataSets._inputs])

    @property
    def labels(self):
        return np.array(self._c_dataSets._labels+self._t_dataSets._labels)
    
    @property
    def seq_lens(self):
        return np.array(self._c_dataSets._seq_lens+self._t_dataSets._seq_lens)
    
    #@property
    def demo(self):
        return np.array([sprs_mat.toarray() for sprs_mat in self._c_dataSets._demo+self._c_dataSets._demo])
    
    @property
    def epochs_completed(self):
        return self._c_dataSets._epochs_completed, self._t_dataSets._epochs_completed

    def next_batch(self, batch_size, ratio=1):
        t_batch_size = int(batch_size/(1+ratio))
        c_batch_size = batch_size - t_batch_size
        
        c_pids, c_inputs, c_labels, c_seq_lens, c_demo = self._c_dataSets.next_batch(c_batch_size)
        t_pids, t_inputs, t_labels, t_seq_lens, t_demo = self._t_dataSets.next_batch(t_batch_size)
        batch_pids = np.array(c_pids+t_pids)
        batch_inputs = np.array([sprs_mat.toarray() for sprs_mat in c_inputs+t_inputs])
        batch_labels = np.array(c_labels+t_labels)
        batch_seq_lens = np.array(c_seq_lens+t_seq_lens)
        batch_demo = np.array([sprs_mat.toarray() for sprs_mat in c_demo+t_demo])
        
        #return batch_inputs, batch_labels, batch_seq_lens, batch_demo
        return sk.utils.shuffle(batch_pids, batch_inputs, batch_labels, batch_seq_lens, batch_demo)
    
def _get_dataSets(controls, treatments):    
    print("Creating_dataSets")
    control_pid, control_data = controls
    treatment_pid, treatment_data = treatments
    
    class DATASETS(object): pass
    dataSets = DATASETS()

    dataSets.train = __concat_dataSets(c_dataSets=__DataSet(pids=control_pid[0], data=control_data[0]), 
                                       t_dataSets=__DataSet(pids=treatment_pid[0], data=treatment_data[0]))
    dataSets.validation = __concat_dataSets(c_dataSets=__DataSet(pids=control_pid[1], data=control_data[1]), 
                                            t_dataSets=__DataSet(pids=treatment_pid[1], data=treatment_data[1]))
    dataSets.test = __concat_dataSets(c_dataSets=__DataSet(pids=control_pid[2], data=control_data[2]), 
                                      t_dataSets=__DataSet(pids=treatment_pid[2], data=treatment_data[2]))
    return dataSets

    

######################################################################################################
    
    
class CDM_DataSetMaker_PIPE_LINE():
    def __init__(self, filePath, DB_connection_info):
        self.filePath = filePath
        self.conn = get_conn(DB_connection_info)
        
    def _Searching_target(self):
        print("\n[Phase 1]..")
        while(1):
            self.search_item_dict, self.target = _searching_from_dict(self.filePath, ref_dict_filename='CDM_concept_dict.pkl')
            if len(self.search_item_dict)!=0: break
            print("\n\tYOU SHOULD RESET YOUR TARGET..")
        
    def _CDM_table_to_sequence(self, q_age_condition, min_visit, pred_interval_unit, pred_interval, ratio, caliper, method, max_visit_clip):
        print("\n[Phase 0] CDM_table_to_sequence")
        self._Searching_target()
        max_time_step, treat_pid, matched_control_pid = Make_seqCohorts(self.conn, self.search_item_dict, q_age_condition, min_visit, pred_interval_unit, pred_interval, ratio, caliper, method, max_visit_clip)
        return max_time_step, treat_pid, matched_control_pid
    
    def _Making_dataSets(self):
        print("\n[Phase 2]..")
        
        print("\n[Phase 3]..")
        rawSeqs_flatted, code_cnt, self.code_to_id, id_to_code = _get_code_dicts(self.conn, self.filePath)
        
        print("\n[Phase 4]..")
        control_multiHot, control_demo, treatment_multiHot, treatment_demo = _get_pid2visit_multiHot(self.conn, self.max_time_step, self.code_to_id)
        
        print("\n[Phase 5]..")
        print("Splitting data")
        _treatments = _split_data('treat', treatment_multiHot, treatment_demo, n_samples=len(treatment_multiHot))
        #_n_samples = (len(_treatment_data[0][2]) + len(_treatment_data[1][2]) + len(_treatment_data[2][2])) * ratio
        #_control_data = _split_data(control_multiHot, control_demo, n_samples=_n_samples)
        _controls = _split_data('control', control_multiHot, control_demo, n_samples=len(control_multiHot)) # 만약 매칭된 control n 수 적으면?
        
        print("\n[Phase 6]..")
        dataSets = _get_dataSets(_controls, _treatments)
        return dataSets, rawSeqs_flatted
    
    def Run_pipeLine(self, q_age_condition=False, min_visit=2, pred_interval_unit='year', pred_interval=1, ratio=4, caliper=0.05, method='PSM', max_visit_clip=20):
        import time
        
        st_time = time.time()
        self.max_time_step, self.treat_pid, self.matched_control_pid = self._CDM_table_to_sequence(q_age_condition, min_visit, pred_interval_unit, pred_interval, ratio, caliper, method, max_visit_clip)
        self.dataSets, self._rawSeqs_flatted = self._Making_dataSets()
        print("[@@] ALL DONE!! at.. {}".format(time.time()-st_time))
        return self.dataSets
    
    def Build_embedding_model(self, left_context_size=4, right_context_size=4, NEW_COOCCUR=False, batch_size=128*2, embedding_size=128, learning_rate=5e-4, decay_steps=2000, decay_rate=0.96, train_steps=200000):
        from glove import glove_ops
        
        self.gloVe_flag = glove_ops.Flag(self.filePath, self.target, self._rawSeqs_flatted, left_context_size, right_context_size, self.code_to_id, NEW_COOCCUR, batch_size=batch_size, embedding_size=embedding_size, learning_rate=learning_rate, decay_steps=decay_steps, decay_rate=decay_rate, checkpoint_dir='checkpoint_gloVe', train_steps=train_steps, printBy=2000, saveBy=10000, new_game=True)
        self.gloVe_pipeLine = glove_ops.gloVe_PIPE_LINE(self.gloVe_flag)
        
    def Run_embedding_model(self, new_game, train_steps=200000, printBy=2000, saveBy=10000):
        self.gloVe_pipeLine.run_graph(new_game, train_steps, printBy, saveBy)
        
        
        