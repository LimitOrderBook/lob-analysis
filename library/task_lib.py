# Used for parallel execution of fit and simulation tasks

import sys
import subprocess
from pymongo import MongoClient, UpdateMany, UpdateOne, InsertOne
from bson import ObjectId
import fit_lib as fl
from contextlib import redirect_stdout
import io
import traceback
import json
import numpy as np
import scipy.optimize

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, bytes):
            return obj.decode('ascii')
        if isinstance(obj,scipy.optimize.LbfgsInvHessProduct):
            return '<omitted>'
        return json.JSONEncoder.default(self, obj)
    
tbl = None

def dbconnect(tbln='tasks', reset=False):
    global tbl
    if tbl is not None and not reset:
        return tbl
    
    url='mongodb://192.168.0.94:27017/'
    client = MongoClient(url)
    tbl = client['global'][tbln]
    return tbl

def add_task(task, execute_sync=False):
    tbl = dbconnect()
    oid = tbl.insert_one({'status':0,'task':task}).inserted_id
    
    if (execute_sync):
        return execute_task_id(oid, force=True)
    return oid
    
    
def execute_task(task):
    if task['task'] == 'fit':
        return fl.do_fit(task)
    if task['task'] == 'simulate':
        return fl.do_simulate(task)
    if task['task'] == 'covariance':
        return fl.do_covariance(task)
    return 'Task not supported.'

def execute_tasks(debug=False):
    tbl = dbconnect()
    tasks = list(
        tbl.aggregate([{"$match":{"status":0}},
                       {"$project":{"_id":1}}])
    )
    for task in tasks:
        tbl.update_one({'_id':task['_id']}, {'$set':{'status':1}}, upsert=False)
        op = ["python3",__file__,str(task['_id'])]
        print(' '.join(op))
        if not debug:
            subprocess.Popen(op)
        
def execute_task_id(oid, force=False):
    if not isinstance(oid, ObjectId):
        oid = ObjectId(oid)
    
    tbl = dbconnect()
    print('Executing taskid ',oid)
    match = {'_id':oid}
    if not force:
        match["status"] = 1
    tasks = list(
        tbl.aggregate([{"$match":match},
                       {"$project":{"_id":1,'task':1}}])
    )
    assert len(tasks) == 1, 'Task not found'
    task = tasks[0]
    
    tbl.update_one({'_id':task['_id']}, {'$set':{'status':2}}, upsert=False)
    
    result = None
    error = None

    f = io.StringIO()
    with redirect_stdout(f):
        try:
            result = execute_task(task['task'])
        except:
            error = traceback.format_exc()
    s = f.getvalue()
    
    tbl.update_one({'_id':task['_id']}, {'$set':{'status':3,'result':json.loads(json.dumps(result, cls=NumpyEncoder)), 'error':error, 'printouts':s}}, upsert=False)
        
if __name__ == "__main__":
    print('started')
    taskid = sys.argv[1]
    assert len(taskid) > 10, 'Invalid TaskId'
      
    execute_task_id(taskid)
    