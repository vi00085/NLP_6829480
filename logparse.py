#Listing log data

import re
from datetime import datetime
def parse_log_file(log_file_path):
    with open(log_file_path, 'r') as file:
        logs = file.readlines()
    log_pattern = re.compile(r'^(?P<timestamp>[\d-]+ [\d:]+,\d+) - Input: (?P<input>.+) \| Predictions: (?P<predictions>.+)$')
    parsed_logs = []
    for log in logs:
        match = log_pattern.match(log)
        if match:
            log_entry = {
                'timestamp': datetime.strptime(match.group('timestamp'), '%Y-%m-%d %H:%M:%S,%f'),
                'input': match.group('input'),
                'predictions': eval(match.group('predictions'))
            }
            parsed_logs.append(log_entry)
    return parsed_logs
log_file_path = 'app.log'
parsed_logs = parse_log_file(log_file_path)
for log in parsed_logs:
    print(log)
