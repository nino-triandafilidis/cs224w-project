"""Parse SEC IAPD individual-feed snapshots into flat tables.

Usage: python analysis/iapd_parse.py [raw_dir] [out_dir]
  raw_dir default data/iapd/raw (one .zip per snapshot, each holding XML parts)
  out_dir default data/iapd/parsed/<snapshot date>/

Tables per snapshot (gzip CSV):
  indvl.csv.gz     indvl_pk, n_drp, has_termination, has_cust_comp
  crnt_emp.csv.gz  indvl_pk, org_pk, org_nm, city, state, first_reg_dt (earliest current registration start)
  branch.csv.gz    indvl_pk, org_pk, city, state, postl_cd (current branch offices)
  prev_reg.csv.gz  indvl_pk, org_pk, org_nm, begin_dt, end_dt
  exam.csv.gz      indvl_pk, exam_cd, exam_dt
"""
import csv
import gzip
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

RAW = Path(sys.argv[1] if len(sys.argv) > 1 else "data/iapd/raw")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "data/iapd/parsed")

TABLES = {
    "indvl": ["indvl_pk", "n_drp", "has_termination", "has_cust_comp"],
    "crnt_emp": ["indvl_pk", "org_pk", "org_nm", "city", "state", "first_reg_dt"],
    "branch": ["indvl_pk", "org_pk", "city", "state", "postl_cd"],
    "prev_reg": ["indvl_pk", "org_pk", "org_nm", "begin_dt", "end_dt"],
    "exam": ["indvl_pk", "exam_cd", "exam_dt"],
}


def gen_date(zf):
    """Snapshot date from the GenOn attribute of the first XML part."""
    with zf.open(zf.namelist()[0]) as f:
        head = f.read(400).decode("latin-1")
    return head.split('GenOn="')[1][:10]


def parse_indvl(el, w):
    info = el.find("Info")
    pk = info.get("indvlPK")
    drps = el.findall("DRPs/DRP")
    w["indvl"].writerow([pk, len(drps),
                         int(any(d.get("hasTermination") == "Y" for d in drps)),
                         int(any(d.get("hasCustComp") == "Y" for d in drps))])
    for emp in el.findall("CrntEmps/CrntEmp"):
        org = emp.get("orgPK")
        starts = [r.get("stDt") for r in emp.findall("CrntRgstns/CrntRgstn") if r.get("stDt")]
        w["crnt_emp"].writerow([pk, org, emp.get("orgNm"), emp.get("city"), emp.get("state"),
                                min(starts) if starts else ""])
        for b in emp.findall("BrnchOfLocs/BrnchOfLoc"):
            w["branch"].writerow([pk, org, (b.get("city") or "").upper().strip(),
                                  b.get("state"), (b.get("postlCd") or "")[:5]])
    for r in el.findall("PrevRgstns/PrevRgstn"):
        w["prev_reg"].writerow([pk, r.get("orgPK"), r.get("orgNm"),
                                r.get("regBeginDt"), r.get("regEndDt")])
    for x in el.findall("Exms/Exm"):
        w["exam"].writerow([pk, x.get("exmCd"), x.get("exmDt")])


def parse_snapshot(zpath):
    zf = zipfile.ZipFile(zpath)
    date = gen_date(zf)
    out = OUT / date
    if (out / "_done").exists():
        return date, "cached"
    out.mkdir(parents=True, exist_ok=True)
    files = {t: gzip.open(out / f"{t}.csv.gz", "wt", newline="") for t in TABLES}
    w = {t: csv.writer(f) for t, f in files.items()}
    for t, cols in TABLES.items():
        w[t].writerow(cols)
    n = 0
    for part in zf.namelist():
        with zf.open(part) as f:
            for _, el in ET.iterparse(f, events=("end",)):
                if el.tag == "Indvl":
                    parse_indvl(el, w)
                    n += 1
                    el.clear()
    for f in files.values():
        f.close()
    (out / "_done").write_text(f"{zpath.name}\n{n}\n")
    return date, n


if __name__ == "__main__":
    for z in sorted(RAW.glob("*.zip")):
        try:
            print(z.name, *parse_snapshot(z), flush=True)
        except Exception as e:  # keep going; report the bad snapshot
            print(z.name, "FAILED", repr(e), flush=True)
