import requests

# Safe sundbox for agent using Docker and tiny image
def run_code(code, output_name, last_step):
    r = requests.post(
        "http://localhost:8000/run_python",
        json={"code": code, "output_name": output_name, "last_step": last_step}
    )
    return r.json()

def run_sql(code, output_name, last_step):
    r = requests.post(
        "http://localhost:8000/run_sql",
        json={"code": code, "output_name": output_name, "last_step": last_step}
    )
    return r.json()

def reset_short_memory():
    r = requests.post(
        "http://localhost:8000/reset"
    )
    return r.json()