import json 
import argparse
def parse_config():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--run_name', type=str, default='dummy')
    arg_parser.add_argument('--config_file', type=str)
    args = arg_parser.parse_args()
    run_name = args.run_name
    config_file = args.config_file
    
    with open(config_file, 'rb') as f:
        json_bytes = f.read()
        json_str = json_bytes.decode('utf-8')
        config_dict = json.loads(json_str)
    print(config_dict)
    #config = TrainConfig.json_deserialize(config_dict)
    #training_setting = TrainSetting(run_name, config)
    #return training_setting

if __name__=='__main__':
    parse_config()