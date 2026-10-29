"""Builds a fictional company: departments, teams, headcounts, level bands and
a roster of fake employees (names, IDs, emails) used to plant PII in synthetic
feedback. Deterministic (fixed seed) so every teammate gets the same company.

    python -m scripts.build_company
Output: data/company.json
"""
import json
import random

from app.config import DATA_DIR

SEED = 42
COMPANY = "Nimbus Technologies Pvt. Ltd. (fictional)"
LOCATIONS = ["Pune", "Bengaluru", "Mumbai", "Gurugram", "Hyderabad"]

# department -> list of (team name, headcount)
STRUCTURE = {
    "Engineering": [("Payments", 18), ("Platform", 22), ("Mobile", 14), ("Data", 12), ("QA", 9)],
    "Sales": [("Enterprise Sales", 16), ("Inside Sales", 20), ("Channel Partners", 7)],
    "Operations": [("Supply Chain", 15), ("Facilities", 8), ("Logistics", 13)],
    "Customer Support": [("Tier 1 Support", 24), ("Escalations", 6)],
    "Marketing": [("Brand", 6), ("Growth Marketing", 9)],
    "Finance": [("Accounts", 10), ("Procurement", 6)],
    "Product": [("Product Management", 11), ("Design", 8)],
    "Human Resources": [("HR Business Partners", 6), ("Talent Acquisition", 7)],
}
PROJECTS = ["Project Falcon", "Project Orion", "Atlas Migration", "Phoenix Revamp",
            "Project Indigo", "Zenith Launch", "Project Monsoon", "Helios Rollout"]

FIRST = ["Rahul", "Priya", "Amit", "Sneha", "Vikram", "Ananya", "Rohan", "Kavya", "Arjun",
         "Neha", "Siddharth", "Pooja", "Karan", "Ishita", "Manish", "Divya", "Aditya", "Meera",
         "Nikhil", "Shreya", "Varun", "Ritu", "Harsh", "Tanvi", "Sameer", "Aisha", "Deepak",
         "Lakshmi", "Farhan", "Gurpreet", "Joseph", "Swati", "Abhishek", "Nandini", "Imran"]
LAST = ["Sharma", "Iyer", "Patel", "Reddy", "Kulkarni", "Nair", "Gupta", "Singh", "Mehta",
        "Banerjee", "Joshi", "Khan", "Menon", "Desai", "Rao", "Chatterjee", "Verma", "Pillai",
        "Fernandes", "Agarwal", "Bhat", "Saxena", "Mishra", "Kapoor", "D'Souza"]


def main() -> None:
    rng = random.Random(SEED)
    used_names, employees, departments = set(), [], []
    emp_no = 10000

    def new_person(dept, team, band, location):
        nonlocal emp_no
        while True:
            name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
            if name not in used_names:
                used_names.add(name)
                break
        emp_no += rng.randint(1, 9)
        first, last = name.lower().split(" ", 1)
        person = {
            "name": name,
            "employee_id": f"NB-{emp_no}",
            "email": f"{first}.{last.replace(chr(39), '')}@nimbustech.example",
            "phone": f"+91 9{rng.randint(100000000, 999999999)}",
            "department": dept, "team": team, "level_band": band, "location": location,
        }
        employees.append(person)
        return person

    for dept, teams in STRUCTURE.items():
        head = new_person(dept, None, "senior_leadership", rng.choice(LOCATIONS))
        team_list = []
        for team, size in teams:
            loc = rng.choice(LOCATIONS)
            manager = new_person(dept, team, "manager", loc)
            for _ in range(size - 1):
                new_person(dept, team, "ic", loc)
            team_list.append({"name": team, "location": loc, "headcount": size,
                              "manager": manager["name"]})
        departments.append({"name": dept, "head": head["name"], "teams": team_list,
                            "headcount": sum(t["headcount"] for t in team_list) + 1})

    company = {
        "company": COMPANY,
        "locations": LOCATIONS,
        "projects": PROJECTS,
        "departments": departments,
        "employees": employees,
        "totals": {
            "employees": len(employees),
            "by_band": {b: sum(e["level_band"] == b for e in employees)
                        for b in ("ic", "manager", "senior_leadership")},
        },
    }
    out = DATA_DIR / "company.json"
    out.write_text(json.dumps(company, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}  ({len(employees)} employees, {len(departments)} departments)")
    print("By level:", company["totals"]["by_band"])


if __name__ == "__main__":
    main()
