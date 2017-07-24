import boto3.session
import sys

MY_INSTANCE_ID = 'xxxx'
MY_INSTANCE_TAG = 'tag:xxx'
MY_INSTANCE_TAG_VALUE = 'xxx'

def get_my_instance(client):
    return client.describe_instances(InstanceIds=[MY_INSTANCE_ID],
                                     Filters=[{'Name': MY_INSTANCE_TAG, 'Values':[MY_INSTANCE_TAG_VALUE]}])

def is_running(my_instance):
    for instance_list in my_instance['Reservations']:
        if len(instance_list['Instances']) != 1:
            return False
        item = instance_list['Instances'][0]
        return item['State']['Name']

def state():
    client = boto3.client('ec2')
    my_instance = get_my_instance(client)
    state = is_running(my_instance)
    print("It's "+ state)

def start():
    client = boto3.client('ec2')
    my_instance = get_my_instance(client)
    if is_running(my_instance) == "stopped":
        client.start_instances(InstanceIds=[MY_INSTANCE_ID])
    else:
        print("it's Not a state that can be stopped")

def stop():
    client = boto3.client('ec2')
    client.stop_instances(InstanceIds=[MY_INSTANCE_ID])

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage : aws.py state|stop|start")
        exit(0)

    option = sys.argv[1]

    if option == "state":
        state()
    elif option == "stop":
        stop()
    elif option == "start":
        start()
    else:
        print("invalid option")


