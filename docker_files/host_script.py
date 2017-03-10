import docker
import os
import sys

def main():
    # Load input args from config file

    # Install docker
    os.system("pip3 install docker")


    # Load Docker-Machine client
    client = docker.from_env()
    container = client.containers.run("ubuntu", "echo hello world")
    print container

    # Build client machines from input args

    # Set Docker secrets for each machine

    for line in container.logs(stream=True):
        print line.strip()
    container.stop()

if __name__ == "__main__":
    main()
    print("Output:", sys.stdout)
