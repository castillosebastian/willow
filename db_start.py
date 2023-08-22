from pymongo import MongoClient, DESCENDING
import subprocess

def db_start():
    try:
        command = "mongod --dbpath /home/sebacastillo/willow/mongodb &"
        subprocess.Popen(command, shell=True)
        print("MongoDB server started successfully!")
    except Exception as e:
        print(f"Failed to start MongoDB server: {e}")


if __name__ == "__main__":
    db_start()