#!/usr/bin/ python3
import os
import sys
import importlib.util
# Constants
BASE_URL = 'unix://var/run/docker.sock'
DOCKERFILE = os.path.join(CWD, 'Dockerfile_TF10_Nvidia.gpu')

spec = importlib.util.spec_from_file_location("docker", "/usr/bin/docker/")
docker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(docker)

# from distutils.sysconfig import get_python_lib

def main():
    # Load input args from secrets (Loaded by host server)


    # Load Docker client - Hopefully nvidia docker?
    client = docker.DockerClient(base_url=BASE_URL)

    # Build dockerfile
    cust_img = images.build(path=DOCKERFILE, rm=True, stream=True)

    # Run dockerfile commands - from Host computer

    # Run built container
    container = client.containers.run(cust_img, command="sh /run.sh")



    print(container)
    for line in container.logs(stream=True):
        print(line.strip())
    container.stop()

if __name__ == "__main__":

    main()
    print("Output:", sys.stdout)
