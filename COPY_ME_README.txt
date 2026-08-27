Revision-2 additions for Edge-Disease-Inference-Engine
======================================================

24 files. Everything else in the repository is unchanged.

WHERE THEY GO
-------------
Copy the contents of this folder into

    C:\Users\Lim Ding Shan\Desktop\Durian project and paper\second paper\Edge-Disease-Inference-Engine-main

merging the folders. Two files are overwritten:

    README.md          rewritten header, new "What changed in revision 2"
                       section, updated Layout / Quick start / Data availability
    requirements.txt   scipy added

Everything else is new:

    analysis\          6 scripts + 5 results_*.json      (new folder)
    audit\phash_distribution.py                          (new file)
    figures\fig8..fig12  PNG + PDF, 10 files             (new files)

THEN
----
Open PowerShell in the repository folder and run:

    git add .
    git status
    git commit -m "Revision 2: threshold, coverage, event-inference and camera analyses; Figs. 8-12"
    git push
    git tag -a v2.0-revision2 -m "Manuscript revision 2 as submitted"
    git push origin v2.0-revision2

Check `git status` before committing. It should list 24 files: 2 modified,
22 new. If it lists hundreds, something in data\ was previously untracked --
stop and check .gitignore before pushing.

CHECK IT WORKS
--------------
    pip install scipy
    python analysis\c12_field_event_sensitivity.py --field data\field_test.csv --lab data\thermal_telemetry --lab data\thermal_telemetry_aug2026

The last lines should report an inferred total of 27.
