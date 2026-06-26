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

def reset_short_memory(): ## not needed in the current system, but could be usefull when Docker container live all the time
    r = requests.post(
        "http://localhost:8000/reset"
    )
    return r.json()