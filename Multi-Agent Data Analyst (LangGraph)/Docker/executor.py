# executor.py
# commented sections are necessary for my current workload. You can easily ignore this if you are not using Impala
# before start setup sql section as you need, do not forget to install required libraries for your setup

from fastapi import FastAPI
from pydantic import BaseModel
import traceback
import sys
import io
#import jaydebeapi
#import jpype
import pandas as pd

app = FastAPI()

SHORT_MEMORY = {}

class CodeRequest(BaseModel):
    code: str
    output_name: str
    last_step: bool

@app.post("/run_python")
def run_code(req: CodeRequest):
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()

    try:
        exec(req.code, SHORT_MEMORY)
        if req.last_step == True:
            return {
                    "status": "success",
                    "output": "success"
                    }
        else:
            res_var = SHORT_MEMORY.get(req.output_name, "No result")
            if str(res_var) == 'No result':
                return {
                    "status": "success",
                    "output": 'No result'
                }
            else:
                try:
                    res_var.info()
                except:
                    print('success')
                output = redirected_output.getvalue()
                return {
                    "status": "success",
                    "output": str(output)
                }
    except Exception:
        return {
            "status": "error exception",
            "output": str(traceback.format_exc())
        }
    finally:
        sys.stdout = old_stdout

@app.post("/run_sql")
def run_code(req: CodeRequest):
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    #jar_path = "/app/ImpalaJDBC41.jar"
    try:
        # if not jpype.isJVMStarted():
        #     jpype.startJVM(
        #         jpype.getDefaultJVMPath(),
        #         f"-Djava.class.path={jar_path}",
        #         "-Djavax.net.debug=ssl:handshake",
        #         "-Djdk.tls.client.protocols=TLSv1.2", # Use only TLS 1.2
        #         "-Dhttps.protocols=TLSv1.2"
        #     )
        # conn = jaydebeapi.connect(
        #     "com.cloudera.impala.jdbc.Driver",
        #     "jdbc:impala://*sensitive_information",
        #     jars=jar_path
        # )
        conn = your_connection
        cursor = conn.cursor()
        cursor.execute(req.code)
        cols = [str(d[0]) for d in cursor.description]
        df_query = pd.DataFrame.from_records(cursor.fetchall(), columns=cols)
        SHORT_MEMORY[req.output_name] = df_query
        df_query.info()
        output = redirected_output.getvalue()
        return{"status": "success", "output": str(output)}
        
    except Exception as e:
       return{"status": "error", "output": str(e)}


@app.post("/reset")
def reset():
    SHORT_MEMORY.clear() ## for new iterations
    return {"status": "reset"}