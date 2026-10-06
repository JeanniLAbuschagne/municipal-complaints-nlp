"""
Generate a synthetic but realistic corpus of municipal complaint narratives.

This exists ONLY so the repository runs end-to-end out of the box without an
external download. For a real submission, replace data/sample_complaints.csv
with the CFPB Consumer Complaint Database (or a 311 service-request export) and
point config.DATA_PATH / config.TEXT_COLUMN at it.
"""
import random
import csv
import os

random.seed(42)

# Each theme = (subject phrases, problem phrases, request phrases)
THEMES = {
    "waste_collection": (
        ["the garbage bin", "our recycling collection", "the green waste pickup",
         "the wheelie bin", "the household refuse service", "the weekly rubbish collection"],
        ["was not collected again this week", "has been skipped three times this month",
         "is overflowing and attracting rats", "was left scattered across the street",
         "keeps getting missed by the truck", "smells terrible and was never emptied"],
        ["please send a crew to empty it", "I want a refund on the service charge",
         "this needs to be fixed urgently", "the collection schedule must be reliable"],
    ),
    "roads_lighting": (
        ["a deep pothole on Main Road", "the broken street light on the corner",
         "the cracked pavement outside my house", "the faded road markings near the school",
         "the damaged kerb on Oak Avenue", "the blocked storm drain on the road"],
        ["has caused damage to my car tyre", "makes the area dangerous at night",
         "is a serious trip hazard for pedestrians", "has been reported but never repaired",
         "floods every time it rains", "needs resurfacing immediately"],
        ["please dispatch a repair team", "I request urgent road maintenance",
         "this is a safety risk and must be addressed", "the street lighting should be restored"],
    ),
    "water_drainage": (
        ["the water supply to my property", "a burst pipe on the sidewalk",
         "the sewage backing up in the drain", "the constant leak near the meter",
         "the low water pressure in the area", "the contaminated tap water"],
        ["has been interrupted for several days", "is wasting thousands of litres",
         "is flooding the lower street", "smells foul and is unsanitary",
         "left us without water all weekend", "appears discoloured and unsafe"],
        ["please send maintenance to repair it", "I demand the leak be stopped",
         "the water service must be restored", "this needs immediate attention"],
    ),
    "billing_rates": (
        ["my municipal account", "the property rates invoice", "the latest utility bill",
         "the electricity charge on my statement", "the water billing amount",
         "the service fee I was charged"],
        ["is far higher than usual with no explanation", "contains charges I never incurred",
         "was estimated incorrectly for months", "doubled without any notice",
         "shows a payment that was never credited", "is clearly a billing error"],
        ["please correct the account and refund me", "I want an itemised statement",
         "the overcharge must be reversed", "this dispute needs to be resolved"],
    ),
    "parking_traffic": (
        ["the parking permit system", "an unfair parking fine", "the new traffic restrictions",
         "the lack of disabled parking bays", "the towing of my legally parked car",
         "the confusing parking signs downtown"],
        ["issued me a penalty I do not deserve", "makes it impossible to park near my home",
         "was applied without proper signage", "ignores residents with permits",
         "caused me to be fined wrongly", "are inconsistent and poorly marked"],
        ["please cancel the unjust fine", "I am appealing this penalty",
         "residents need fair parking access", "the signage must be made clear"],
    ),
    "noise_nuisance": (
        ["construction noise next door", "the late-night disturbance from the bar",
         "the loud parties in the neighbourhood", "the constant barking from a property",
         "the early morning roadworks", "the illegal dumping on the empty lot"],
        ["continues well past permitted hours", "makes it impossible to sleep",
         "violates the local noise bylaw", "has been reported repeatedly with no action",
         "starts before sunrise every day", "is creating a public health hazard"],
        ["please enforce the noise regulations", "I want an inspector to investigate",
         "the bylaw must be enforced", "this nuisance needs to stop"],
    ),
    "parks_recreation": (
        ["the local playground equipment", "the public park near the river",
         "the community sports field", "the picnic area in the green belt",
         "the walking trail behind the estate", "the public swimming pool"],
        ["is broken and unsafe for children", "is covered in litter and overgrown",
         "has not been maintained in months", "has vandalised and dangerous equipment",
         "is closed without any explanation", "lacks basic lighting and security"],
        ["please repair and clean the facility", "the park needs regular maintenance",
         "families deserve a safe space", "I request restoration of the amenity"],
    ),
    "public_transport": (
        ["the bus service on my route", "the timetable at the local stop",
         "the shuttle to the station", "the accessibility of the bus stop",
         "the frequency of the morning buses", "the condition of the bus shelter"],
        ["is consistently late or cancelled", "no longer matches the posted schedule",
         "skips my stop during peak hours", "is not wheelchair accessible",
         "leaves commuters stranded for hours", "is broken and offers no shelter"],
        ["please restore a reliable service", "the schedule must be honoured",
         "commuters need dependable transport", "I request an improved timetable"],
    ),
}

CONNECTORS = [
    "I am writing to complain that", "I wish to formally report that",
    "It is unacceptable that", "I have repeatedly raised that",
    "Despite several calls,", "For the third time,",
    "As a ratepayer I must report that", "I am extremely frustrated that",
]

rows = []
for theme, (subjects, problems, requests) in THEMES.items():
    for _ in range(65):
        text = (f"{random.choice(CONNECTORS)} {random.choice(subjects)} "
                f"{random.choice(problems)}. {random.choice(requests).capitalize()}.")
        rows.append({"complaint_id": len(rows) + 1, "narrative": text, "true_theme": theme})

random.shuffle(rows)

out_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_complaints.csv")
out_path = os.path.abspath(out_path)
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["complaint_id", "narrative", "true_theme"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} synthetic complaints to {out_path}")
