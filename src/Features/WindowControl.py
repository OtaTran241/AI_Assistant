import subprocess
import psutil    

def open_app(app_name):
    try:
        subprocess.Popen(["start", app_name], shell=True)
        print(f"{app_name} opened successfully.")
        return True
    except Exception as e:
        print(f"Failed to open {app_name}: {str(e)}")
        return False

def close_app(app_name):
    for proc in psutil.process_iter(['pid', 'name']):
        if app_name in proc.info['name'].lower():
            pid = proc.info['pid']
            try:
                process = psutil.Process(pid)
                process.terminate()
                print(f"{app_name} closed successfully.")
                return True
            except Exception as e:
                print(f"Failed to close {app_name}: {str(e)}")
                return False
    return False
        