import os
import re
import json
import copy
import pprint
import argparse
from pathlib import Path
from openai import OpenAI
from datetime import datetime, timezone

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_PATH = "/Users/zilinghan/Documents/projects/mma/ai4mma-main/Jy_all"
OUTPUT_JSON = "result/all_GCN_data_gpt55_run1.json"
MODEL = "GPT-5.5"
MAX_RETRY = 3
TEMPERATURE = 1.0
MAX_TOKENS = 10000
# ──────────────────────────────────────────────────────────────────────────────


def read_openai_key(file_path):
    with open(file_path, "r") as f:
        return f.read().strip()


def extract_json(text):
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    json_data = []
    try:
        assert len(matches) > 0
        for match in matches:
            json_data.append(json.loads(match))
        return json_data
    except:
        raise ValueError("No valid JSON found in the text.")


def generate_json_formatted_response(
    prompt: str,
    client: OpenAI,
    expected_keys: list,
    model: str = MODEL,
    max_retry: int = 3,
    temperature: float = 0.0,
    max_tokens: int = 1000,
):
    json_response = None
    while max_retry > 0:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            json_response = response.choices[0].message.content
            json_response_copy = copy.deepcopy(json_response)
            response_list_raw = extract_json(json_response)
            response_list = []
            valid_list_response = True
            for resp in response_list_raw:
                resp = {key.lower(): value for key, value in resp.items()}
                valid_response = True
                for key in resp:
                    if key not in expected_keys:
                        valid_response = False
                        break
                if valid_response:
                    response_list.append(resp)
                else:
                    valid_list_response = False
                    break
            if valid_list_response:
                return response_list
            else:
                print("Invalid response keys, retrying...")
                max_retry -= 1
        except ValueError:
            print(json_response_copy)
            print("Invalid JSON response, retrying...")
            max_retry -= 1
        except Exception as e:
            print(f"Error generating response: {e}")
            break
    return None


def get_txt_filenames(folder_path):
    if not os.path.isdir(folder_path):
        raise ValueError(f"The specified folder does not exist: {folder_path}")
    return sorted(f for f in os.listdir(folder_path) if f.endswith(".txt"))


def find_RA(GCN_text):
    ra_pattern = re.compile(r"\b(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)\b")
    ra_list = []
    for match in ra_pattern.finditer(GCN_text):
        start, _ = match.span()
        search_window = GCN_text[max(0, start - 12):start]
        if re.search(r"RA", search_window, re.IGNORECASE):
            ra_list.append(match.group(1))
    return ra_list


def find_Dec(GCN_text):
    dec_pattern = re.compile(r"-?\d{1,2}:\d{2}:\d{2}(?:\.\d+)?")
    dec_list = []
    for match in dec_pattern.finditer(GCN_text):
        start, _ = match.span()
        search_window = GCN_text[max(0, start - 12):start]
        if re.search(r"Dec", search_window, re.IGNORECASE):
            dec_list.append(match.group(0))
    return dec_list


def extract_gps_time(text):
    match = re.search(
        r"DATE:\s+(\d{2})/(\d{2})/(\d{2}) (\d{2}):(\d{2}):(\d{2}) GMT", text
    )
    if not match:
        raise ValueError("Date format not found in text")
    year, month, day, hour, minute, second = map(int, match.groups())
    year += 2000 if year < 80 else 1900
    dt = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc)
    gps_seconds = int((dt - gps_epoch).total_seconds()) + 18
    return gps_seconds, match[0]


def load_and_process_GCN(
    path_GCN: Path,
    client: OpenAI,
    model: str = MODEL,
    max_retry: int = MAX_RETRY,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
):
    print(f"==== Starting to process {path_GCN} ====")
    all_data_points = []
    GCN_text = path_GCN.read_text(encoding="utf-8")

    GCN_number = None
    for line in GCN_text.splitlines():
        if "NUMBER" in line:
            parts = line.split()
            if len(parts) > 1:
                GCN_number = parts[1]
            break

    gps_seconds, time_string = extract_gps_time(GCN_text)

    prompt1 = (
        "Based on the individual GCN circular provided below, please decide if it does a radio "
        "follow-up on the optical transient counterpart to GW170817. This should be targeted at "
        'SSS17a. Please record "false" if the followup is not done on the optical transient '
        'counterpart and "true" if it is done on the optical transient counterpart (SSS17a)\n'
        "The text of the GCN is as follows:\n"
        "==================================\n"
        + GCN_text
        + "\n==================================\n"
        "Please make your response as a strict JSON formatted string without any additional text. "
        "The JSON should contain a single key-value pair as either:\n"
        '{"optical_transient": "true"}\nor\n{"optical_transient": "false"}\n'
    )

    response1 = generate_json_formatted_response(
        prompt=prompt1,
        client=client,
        expected_keys=["optical_transient"],
        model=model,
        max_retry=max_retry,
        temperature=temperature,
        max_tokens=max_tokens,
    )[0]
    print(response1)

    prompt2 = (
        "Based on the individual GCN circular provided below, please decide if it contains any "
        "statement claiming there is no significant emission detected from the optical transient "
        '(called SSS17a). Please record "true" if the GCN includes a non-detection statement '
        "pertaining to the transient. Please record \"false\" if the GCN does not include a "
        "non-detection statement.\n"
        "The text of the GCN is as follows:\n"
        "==================================\n"
        + GCN_text
        + "\n==================================\n"
        "Please make your response as a strict JSON formatted string without any additional text. "
        "The JSON should contain a single key-value pair as either:\n"
        '{"non_emission_statement": "true"}\nor\n{"non_emission_statement": "false"}\n'
    )

    response2 = generate_json_formatted_response(
        prompt=prompt2,
        client=client,
        expected_keys=["non_emission_statement"],
        model=model,
        max_retry=max_retry,
        temperature=temperature,
        max_tokens=max_tokens,
    )[0]
    print(response2)

    if response1["optical_transient"].lower() == "true":
        right_ascentions = find_RA(GCN_text)
        declinations = find_Dec(GCN_text)

        prompt3 = (
            "Based on the individual GCN circular provided below, please help me extract data "
            "that is only pertaining to the optical transient (SSS17a). For observations of the "
            'optical transient, could you please extract the frequencies of observation ("frequency"), '
            'any fluxes that were recorded ("flux_density"), and whether these are observations or '
            'upper limits ("type"). Please include units whenever possible. I would like to have the '
            "frequency in GHz and the flux density in mJy. Remember, 1 GHz = 1000000000 Hz. I also "
            'want you to record the name of the target which is being reported on ("name"), if it is '
            "given. If the target is not named, please do not report a name. Please also record the "
            'right ascension ("right_ascension") and declination ("declination"), if they are given. '
            "If not, please do not report them.\n\n"
            "Please use these conversions to report the flux density in mJy:\n\n"
            "1 uJy = 1 microJy = 1 muJy\n"
            "1000 uJy = 1 mJy = 1 milliJy\n"
            "1000 mJy = 1 Jy\n\n"
            "We do not count emission from NGC 4993, which is the host galaxy.\n"
            "Please only discuss data available in this GCN, not other GCNs.\n\n"
            "The text of the GCN is as follows:\n"
            "==================================\n"
            + GCN_text
            + "\n==================================\n\n"
            "Please make your response as a strict JSON formatted string without any additional text. "
            "The JSON should contain any key-value pairs that are relevant to the optical transient "
            "(SSS17a). For example:\n"
            "{\n"
            '    "frequency": ...,\n'
            '    "flux_density": ...,\n'
            '    "name": ...,\n'
            '    "right_ascension": ...,\n'
            '    "declination": ...,\n'
            '    "type": "upper_limit" | "observation"\n'
            "}.\n\n"
            'The expected keys are: "frequency", "flux_density", "name", "right_ascension", '
            '"declination", and "type". Please make sure accurate units are provided for frequency '
            "flux_density. If certain information is not available, please omit those keys from the "
            "JSON response. Please do not include any other information or context in the response. "
            "The JSON should only contain the relevant data extracted from the GCN.\n\n"
            "Additional rules:\n"
            "- If there are multiple observations (different frequencies, different times, OR "
            "different flux values at the same frequency), return one separate JSON object per "
            "observation. Never put multiple values into a single array-valued field, and never "
            "deduplicate observations that share the same frequency.\n"
            "- Report frequencies exactly as written in the GCN. Do not round or approximate.\n"
            "- Report flux density as a plain numeric value with unit only. "
            "Do not include < or ~ prefixes, and do not append suffixes like /beam or rms.\n"
            "- Always include \"frequency\" and \"flux_density\" if at all possible. "
            "If the flux density value contains extra text or expressions, "
            "still report the numeric part in mJy and ignore the rest.\n"
        )

        response3 = generate_json_formatted_response(
            prompt=prompt3,
            client=client,
            expected_keys=["frequency", "flux_density", "name", "right_ascension", "declination", "type"],
            model=model,
            max_retry=max_retry,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response3 is None:
            print("No valid JSON response from OpenAI.")
            return []

        for json_data in response3:
            # Strip noise-formula suffixes from flux_density, e.g. "1400 mJy (w/64 us)^-1/2"
            if "flux_density" in json_data:
                fd = str(json_data["flux_density"])
                fd = re.sub(r"\s*[\(\^].*", "", fd).strip()  # drop anything from ( or ^ onward
                fd = re.sub(r"\s*(rms|/beam)\b.*", "", fd, flags=re.IGNORECASE).strip()
                json_data["flux_density"] = fd

            json_data["type"] = json_data.get("type", "upper_limit")
            json_data["detected"] = json_data["type"] == "observation"

            non_emission_statement = response2["non_emission_statement"].lower() == "true"
            json_data["non_emission_statement"] = non_emission_statement

            if json_data["non_emission_statement"] and json_data["detected"]:
                print("Warning: Non-emission statement found in GCN.")
                json_data["detected"] = False

            json_data["time"] = str(gps_seconds)
            json_data["GCN_number"] = GCN_number
            json_data["time_string"] = time_string

            if len(right_ascentions) == 0 and len(declinations) == 0:
                json_data["ra"] = None
                json_data["dec"] = None
            elif len(right_ascentions) == 1 and len(declinations) == 1:
                json_data["ra"] = right_ascentions[0]
                json_data["dec"] = declinations[0]
            else:
                prompt4 = (
                    "Below shows the text for an individual GCN circular:\n"
                    "==================================\n"
                    + GCN_text
                    + "\n==================================\n"
                    f"There is an event detected in this GCN which is most closely described by a "
                    f"frequency of {json_data['frequency']} GHz and flux density "
                    f"{json_data['flux_density']} mJy, and comes from the optical transient "
                    "counterpart, called SSS17a. From the lists below, could you please help me "
                    "select the right ascension and declination that best describe the optical "
                    "transient location? Please do not comment on any other galaxies or the host galaxy.\n\n"
                    f"Here is the right ascension list: {right_ascentions}\n"
                    f"Here is the declination list: {declinations}\n\n"
                    "Please make your response as a strict JSON formatted string without any additional text. "
                    'The JSON should contain two keys, "best_right_ascension" and "best_declination". For example:\n'
                    "{\n"
                    '    "best_right_ascension": ...,\n'
                    '    "best_declination": ...\n'
                    "}.\n"
                )
                response4 = generate_json_formatted_response(
                    prompt=prompt4,
                    client=client,
                    expected_keys=["best_right_ascension", "best_declination"],
                    model=model,
                    max_retry=max_retry,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )[0]
                json_data["ra"] = response4["best_right_ascension"]
                json_data["dec"] = response4["best_declination"]

            all_data_points.append(json_data)

    pprint.pprint(all_data_points)
    print(f"==== Finished processing {path_GCN} ====\n")
    return all_data_points


def main():
    parser = argparse.ArgumentParser(description="Extract GCN radio observations via LLM.")
    parser.add_argument("--model",  default=MODEL,       help="Model name (default: %(default)s)")
    parser.add_argument("--output", default=OUTPUT_JSON, help="Output JSON path (default: %(default)s)")
    parser.add_argument("--temp",   default=TEMPERATURE, type=float, help="Sampling temperature (default: %(default)s)")
    args = parser.parse_args()

    client = OpenAI(
        api_key="zilinghan.li",
        base_url="https://apps.inside.anl.gov/argoapi/v1"
    )
    dir_data = Path(DATA_PATH)
    filenames = get_txt_filenames(dir_data)
    print(f"Model      : {args.model}")
    print(f"Temperature: {args.temp}")
    print(f"Output     : {args.output}")
    print(f"Found {len(filenames)} GCN files in {dir_data}\n")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    all_data = []
    for filename in filenames:
        data = load_and_process_GCN(
            path_GCN=dir_data / filename,
            client=client,
            model=args.model,
            max_retry=MAX_RETRY,
            temperature=args.temp,
            max_tokens=MAX_TOKENS,
        )
        all_data.extend(data)

    with open(args.output, "w") as f:
        json.dump(all_data, f, indent=4)
    print(f"\nSaved {len(all_data)} data points to {args.output}")


if __name__ == "__main__":
    main()
