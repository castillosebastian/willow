from pymongo import MongoClient, DESCENDING
import subprocess
import time


def db_stop():
    try:
        command = 'mongo admin --eval "db.shutdownServer()" &'
        subprocess.run(command, shell=True, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        print("MongoDB server stopped successfully!")

        # Optionally, add a delay to ensure that the shutdown has time to take effect
        time.sleep(2)
    except subprocess.CalledProcessError as e:
        print(f"Failed to stop MongoDB server. Error message: {e.stderr.decode()}")


if __name__ == "__main__":
    db_stop()