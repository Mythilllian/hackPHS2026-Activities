"""SkyExplorer: explore live aircraft data from OpenSky. Run: venv/bin/python skyexplorer.py"""
import json, os, sys, time
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
import requests

# Areas: (lat_min, lon_min, lat_max, lon_max)
AREAS = {"United States": (25, -125, 50, -65), "Europe": (35, -12, 72, 40)}
# First 3 letters of a callsign = airline
AIRLINES = {"UAL": "United", "DAL": "Delta", "AAL": "American", "SWA": "Southwest",
            "JBU": "JetBlue", "FDX": "FedEx", "UPS": "UPS", "BAW": "British Airways",
            "DLH": "Lufthansa", "AFR": "Air France", "KLM": "KLM", "RYR": "Ryanair"}


def load(area):
    """Download aircraft for an area. The answer is saved for 30 minutes to save API credits."""
    os.makedirs(".cache", exist_ok=True)
    path = ".cache/" + "_".join(map(str, area)) + ".json"
    if not (os.path.exists(path) and time.time() - os.path.getmtime(path) < 1800):
        creds = json.load(open("credentials.json"))
        token = requests.post(
            "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token",
            data={"grant_type": "client_credentials", "client_id": creds["clientId"],
                  "client_secret": creds["clientSecret"]}).json()["access_token"]
        reply = requests.get("https://opensky-network.org/api/states/all", timeout=60,
                             params=dict(zip(["lamin", "lomin", "lamax", "lomax"], area)),
                             headers={"Authorization": "Bearer " + token})
        reply.raise_for_status()
        json.dump(reply.json(), open(path, "w"))
    planes = []
    for s in json.load(open(path))["states"] or []:
        alt = s[7] * 3.28084 if s[7] is not None else None   # meters -> feet
        if alt is not None and not 0 <= alt <= 60000:
            alt = None                                        # impossible reading: ignore
        call = (s[1] or "").strip()
        planes.append({"call": call, "country": s[2], "lon": s[5], "lat": s[6], "alt": alt,
                       "speed": s[9] * 1.94384 if s[9] is not None else None,  # m/s -> knots
                       "track": s[10], "ground": s[8], "squawk": s[14],
                       "airline": AIRLINES.get(call[:3]) if call[3:4].isdigit() else None})
    return planes


def menu(title, options):
    """Print numbered options and return the number the user types."""
    print("\n" + title)
    for i, o in enumerate(options, 1):
        print(f"  {i}. {o}")
    while True:
        try:
            answer = input("Enter a number: ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit("\nGoodbye.")
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer)
        print("Please type one of the numbers above.")


def flight_levels(planes):
    # separates by level to find west and east bound flights
    east, west = [], []
    for p in planes:
        if p["ground"] or p["alt"] is None or p["track"] is None or not 24000 <= p["alt"] <= 45000:
            continue
        level = round(p["alt"] / 1000)
        if abs(p["alt"] - level * 1000) < 200:
            (east if p["track"] < 180 else west).append(level)
    return east, west


def findings(planes):
    air = [p for p in planes if not p["ground"]]
    print(f"\n{len(planes)} aircraft, {len(air)} in the air, from {len({p['country'] for p in planes})} countries.")
    top = Counter(p["airline"] for p in planes if p["airline"]).most_common(5)
    print("Top airlines:", ", ".join(f"{a} ({n})" for a, n in top))
    east, west = flight_levels(planes)
    if len(east) > 30 and len(west) > 30:
        print(f"\nTHE BIG PATTERN: {100 * sum(l % 2 for l in east) // len(east)}% of eastbound planes fly at ODD flight "
              f"levels (FL350)\nand {100 * sum(l % 2 == 0 for l in west) // len(west)}% of westbound planes at EVEN ones "
              f"(FL360). Air traffic rules keep\nopposite directions at different altitudes. See chart 3.")
    for p in planes:
        if p["squawk"] in ("7500", "7600", "7700"):
            print(f"EMERGENCY CODE {p['squawk']}: {p['call']} ({p['country']})")


def show(planes, title):
    print(f"\n{title}: {len(planes)} aircraft (showing the 15 highest)")
    for p in sorted(planes, key=lambda p: p["alt"] or -1, reverse=True)[:15]:
        print(f"  {p['call'] or '-':<9} {p['country'][:16]:<17} {p['alt'] or 0:>7,.0f} ft  {p['speed'] or 0:>4.0f} kt")


def search(planes):
    word = input("Search callsign, country or airline: ").strip().lower()
    found = [p for p in planes if word and word in f"{p['call']} {p['country']} {p['airline']}".lower()]
    show(found, f"Results for '{word}'") if found else print("No matches.")


def add_filter(filters):
    kind = menu("Filter by", ["Country", "Airline", "Minimum altitude", "Airborne only"])
    if kind == 3:
        try:
            low = float(input("Minimum altitude in feet: "))
        except ValueError:
            return print("That is not a number.")
        filters.append((f"above {low:,.0f} ft", lambda p: p["alt"] is not None and p["alt"] >= low))
    elif kind == 4:
        filters.append(("airborne", lambda p: not p["ground"]))
    else:
        field = "country" if kind == 1 else "airline"
        word = input(f"Type part of a {field} name: ").strip().lower()
        filters.append((f"{field} ~ {word}", lambda p: word in (p[field] or "").lower()))


def compare(planes):
    field = "country" if menu("Compare by", ["Country", "Airline"]) == 1 else "airline"
    names = [n for n, _ in Counter(p[field] for p in planes if p[field]).most_common(5)]
    print(f"\n{field.title():<16} {'Aircraft':>8} {'Avg alt ft':>11} {'Avg knots':>10}")
    for n in names:
        group = [p for p in planes if p[field] == n and not p["ground"] and p["alt"] and p["speed"]]
        if group:
            print(f"{n:<16} {len(group):>8} {sum(p['alt'] for p in group) / len(group):>11,.0f} "
                  f"{sum(p['speed'] for p in group) / len(group):>10.0f}")


def chart(planes):
    kind = menu("Which chart?", ["Map colored by altitude", "Altitude histogram",
                                 "Eastbound vs westbound altitudes", "Speed vs altitude"])
    air = [p for p in planes if not p["ground"] and p["alt"] is not None]
    fig, ax = plt.subplots(figsize=(10, 6))
    if kind == 1:
        dots = ax.scatter([p["lon"] for p in air], [p["lat"] for p in air], c=[p["alt"] for p in air], s=4)
        fig.colorbar(dots, label="Altitude (ft)")
        labels = ["Where aircraft are", "Longitude", "Latitude"]
    elif kind == 2:
        ax.hist([p["alt"] for p in air], bins=40)
        labels = ["Most aircraft fly low or in the cruise layer", "Altitude (ft)", "Aircraft"]
    elif kind == 3:
        east, west = flight_levels(planes)
        levels = range(24, 46)
        ax.bar([l - .2 for l in levels], [east.count(l) for l in levels], .4, label="Eastbound")
        ax.bar([l + .2 for l in levels], [west.count(l) for l in levels], .4, label="Westbound")
        ax.set_xticks(levels)
        ax.legend()
        labels = ["Eastbound and westbound planes use different altitudes", "Flight level (35 = 35,000 ft)", "Aircraft"]
    else:
        ax.scatter([p["alt"] for p in air if p["speed"]], [p["speed"] for p in air if p["speed"]], s=4)
        labels = ["Aircraft fly faster the higher they go", "Altitude (ft)", "Speed (knots)"]
    if input("Type y to change the title and labels, or Enter to skip: ").lower() == "y":
        labels = [input(f"{name} [{old}]: ") or old for name, old in zip(["Title", "X label", "Y label"], labels)]
    ax.set_title(labels[0])
    ax.set_xlabel(labels[1])
    ax.set_ylabel(labels[2])
    fig.savefig("chart.png", dpi=120)
    print("Saved chart.png")
    if matplotlib.get_backend() != "Agg":
        plt.show()


area = list(AREAS)[menu("Where do you want to look?", list(AREAS)) - 1]
everything = load(AREAS[area])
filters = []
while True:
    planes = [p for p in everything if all(test(p) for _, test in filters)]
    print(f"\n=== {area}: {len(planes)} aircraft | filters: {', '.join(d for d, _ in filters) or 'none'} ===")
    choice = menu("Main menu", ["Key findings", "Browse", "Search", "Add a filter", "Clear filters",
                                "Compare", "Charts", "Quit"])
    if choice == 8:
        break
    if choice == 4:
        add_filter(filters)
    elif choice == 5:
        filters.clear()
    elif not planes:
        print("No aircraft match. Choose 'Clear filters'.")
    else:
        {1: findings, 2: lambda p: show(p, "Browse"), 3: search, 6: compare, 7: chart}[choice](planes)
