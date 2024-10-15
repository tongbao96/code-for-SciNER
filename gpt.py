import json
import http.client
import os

input_file_path = '../data/SciERC/test.json'
output_file_path = 'SciERC-ChatGPT-results.json'
progress_file_path = 'progress.txt'
api_key = ''

def process_json_file(input_file_path, output_file_path, progress_file_path, api_key):
    if os.path.exists(progress_file_path):
        with open(progress_file_path, 'r', encoding='utf-8') as progress_file:
            last_processed_index = int(progress_file.read().strip())
    else:
        last_processed_index = -1


    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    total_lines = len(lines)
    conn = http.client.HTTPSConnection("aigc-api-backup.x-see.cn")

    headers = {
        'Authorization': f'sk-{api_key}',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }

    if os.path.exists(output_file_path):
        with open(output_file_path, 'r', encoding='utf-8') as outfile:
            results = json.load(outfile)
    else:
        results = {}

    for i, line in enumerate(lines, 1):
        data = json.loads(line)
        raw_index = data['raw_index']

        if raw_index <= last_processed_index:
            continue

        if data['input'].startswith("Please list all scientific"):
            prompt = data['input']
            payload = json.dumps({
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })

            conn.request("POST", "/v1/chat/completions", payload, headers)
            res = conn.getresponse()
            api_data = res.read().decode("utf-8")
            api_result = json.loads(api_data)

            assistant_message = api_result['choices'][0]['message']['content']

            results[raw_index] = {
                "input": prompt,
                "api_result": assistant_message,
                "target": data.get("target", ""),
                "labels": data.get("labels", [])
            }

            if i % 25 == 0:
                with open(output_file_path, 'w', encoding='utf-8') as outfile:
                    json.dump(results, outfile, ensure_ascii=False, indent=4)

                with open(progress_file_path, 'w', encoding='utf-8') as progress_file:
                    progress_file.write(str(raw_index))

                print(f"Saved progress at line {i}/{total_lines}, raw_index {raw_index}.")

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=4)

    with open(progress_file_path, 'w', encoding='utf-8') as progress_file:
        progress_file.write(str(raw_index))

    print(f"Processing complete. Output written to {output_file_path}")

