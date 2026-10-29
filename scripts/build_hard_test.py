"""Builds a small HAND-WRITTEN hard test set (40 notes) to check the models beyond the easy
synthetic data. Notes avoid obvious keywords, mix topics, use Hinglish, sarcasm or indirect
wording. Labels follow one rule: the category is the MAIN issue the employee wants addressed.
Committee cases (harassment, safety, fraud, hr_conduct) should be escalated.

    python -m scripts.build_hard_test
Output: data/synthetic/hard_test.jsonl
"""
import json

from app.config import DATA_DIR, ESCALATION_ROUTES

# (category, difficulty, text)
NOTES = [
    # compensation
    ("compensation", "vague", "Joined two years ago and I'm still taking home what new joiners are being offered now. Feels odd."),
    ("compensation", "no_keywords", "The numbers on my slip haven't matched what was promised in my offer letter since April, and nobody explains why."),
    ("compensation", "hinglish", "Q3 ka payout abhi tak nahi aaya, aur ab bol rahe hain ki target hi galat set hua tha."),
    ("compensation", "sarcasm", "Love how an 'exceptional' rating came with a 2% hike. Truly motivating stuff."),
    # workload
    ("workload", "no_keywords", "I haven't had a proper lunch break in weeks. Every day ends after the building lights go off."),
    ("workload", "indirect", "Since two people left, nobody was replaced and their stuff just landed on the rest of us."),
    ("workload", "hinglish", "Sunday ko bhi laptop kholna padta hai, raat ko 11 baje tak pings aate rehte hain."),
    ("workload", "mixed", "Honestly the pay is fine, it's just that I now do three people's jobs."),
    # manager_behaviour
    ("manager_behaviour", "no_keywords", "My lead presents my work as his own in every review and corrects me in front of clients."),
    ("manager_behaviour", "no_keywords", "He only gives the good projects to the people who go on smoke breaks with him."),
    ("manager_behaviour", "vague", "Things have just been off with my reporting manager lately. I dread our one-on-ones."),
    ("manager_behaviour", "hinglish", "Boss ne poori team ke saamne mujhe bewakoof bola, aur baad mein bola mazaak tha."),
    # culture
    ("culture", "no_keywords", "Nobody says thanks around here. You ship something big and the next day it's like it never happened."),
    ("culture", "indirect", "The main office gets every town hall and celebration; those of us in the smaller office feel like an afterthought."),
    ("culture", "no_keywords", "Lunch tables are basically split by language, and new people end up eating alone."),
    ("culture", "sarcasm", "Great place to work if you enjoy WhatsApp groups you're not invited to."),
    # growth
    ("growth", "vague", "Third cycle in a row I've been told 'next time'. I still don't know what next time requires."),
    ("growth", "mixed", "I applied for an open role in another team and my manager quietly blocked the move."),
    ("growth", "mixed", "My training budget hasn't been used in two years because there's never any time for it."),
    ("growth", "hinglish", "Senior banne ke liye kya karna padega, yeh koi batata hi nahi."),
    # harassment (committee)
    ("harassment", "no_keywords", "He keeps standing too close at the coffee machine and 'jokingly' touches my shoulder. I've started taking the stairs to avoid him."),
    ("harassment", "indirect", "My lead texts me late at night asking what I'm wearing and says it's just banter."),
    ("harassment", "mixed", "Ever since I turned down his invitation for drinks, my reviews got worse and he talks about my 'attitude' in front of everyone."),
    ("harassment", "hinglish", "Woh senior har meeting ke baad mujhe akele milne bolta hai aur meri photos pe ajeeb comments karta hai."),
    # safety (committee)
    ("safety", "no_keywords", "The back door near the loading bay is chained shut once the night shift starts."),
    ("safety", "indirect", "Late-night cab drivers keep changing; last week the driver switched off the GPS and took a different route home."),
    ("safety", "no_keywords", "There's been a burning smell in the server room for two days and nobody has checked it."),
    ("safety", "hinglish", "Forklift bina horn ke chal raha hai warehouse mein, kal ek banda bal bal bacha."),
    # fraud (committee)
    ("fraud", "no_keywords", "My lead asked me to put his family dinner on the client entertainment claim."),
    ("fraud", "indirect", "We were told to mark tickets as resolved before customers confirm, so the monthly numbers look better."),
    ("fraud", "indirect", "The same supplier wins every bid, and it happens to be the purchase head's brother-in-law."),
    ("fraud", "hinglish", "Attendance sheet mein aise logon ka naam bhi daal rahe hain jo office aate hi nahi."),
    # hr_conduct (committee)
    ("hr_conduct", "no_keywords", "I told the people team about it in confidence, and within a day my manager knew every detail."),
    ("hr_conduct", "vague", "Raised a grievance with my HRBP in March. It was closed in April without a single reply."),
    ("hr_conduct", "indirect", "The HRBP advised me to 'think about my career' before filing anything against a director."),
    ("hr_conduct", "hinglish", "HR wale complaint lene se pehle hi bol dete hain ki corporate mein aisa hi hota hai."),
    # other
    ("other", "no_keywords", "Can we get a second microwave? The queue at 1 pm is twenty minutes long."),
    ("other", "no_keywords", "The VPN drops every hour and support keeps saying the problem is on my side."),
    ("other", "no_keywords", "Parking has been chaos since the new block opened; people keep blocking each other in."),
    ("other", "suggestion", "Suggestion: a small quiet room for people who need to take personal calls during the day."),
]


def main() -> None:
    out = DATA_DIR / "synthetic" / "hard_test.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for i, (cat, diff, text) in enumerate(NOTES):
            f.write(json.dumps({"id": f"hard-{i:03d}", "category": cat, "difficulty": diff,
                                "escalate": cat in ESCALATION_ROUTES, "text": text}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(NOTES)} hard test notes to {out}")


if __name__ == "__main__":
    main()
