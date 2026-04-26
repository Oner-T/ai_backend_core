"""
Load GroundTruthEntry rows from golden_dataset.json.

Usage:
    python manage.py load_ground_truth
    python manage.py load_ground_truth --file path/to/other.json
    python manage.py load_ground_truth --dry-run   # validate only, don't save
    python manage.py load_ground_truth --reset      # clear existing entries first

Expected JSON format:
[
  {
    "question": "Kişisel veri nedir?",
    "expected_answer": "...",          # optional
    "scenario": "definition",          # optional, defaults to "user_validated"
    "expected_sources": [
      {"document_name": "KVKK", "madde": 3}
    ]
  }
]
"""

import json
import os

from django.core.management.base import BaseCommand, CommandError

from intelligence.models import GroundTruthEntry

VALID_SCENARIOS = {s[0] for s in GroundTruthEntry.SCENARIO_CHOICES}
DEFAULT_FILE    = os.path.join(os.getcwd(), "golden_dataset.json")
DIVIDER         = "─" * 70


class Command(BaseCommand):
    help = "Import GroundTruthEntry rows from golden_dataset.json"

    def add_arguments(self, parser):
        parser.add_argument(
            '--file', type=str, default=DEFAULT_FILE,
            help=f'Path to JSON file (default: {DEFAULT_FILE})'
        )
        parser.add_argument(
            '--dry-run', action='store_true',
            help='Validate and report without writing to DB'
        )
        parser.add_argument(
            '--reset', action='store_true',
            help='Delete all existing GroundTruthEntry rows before importing'
        )

    def handle(self, *args, **options):
        path    = options['file']
        dry_run = options['dry_run']
        reset   = options['reset']

        if not os.path.exists(path):
            raise CommandError(f"File not found: {path}")

        with open(path, encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise CommandError(f"Invalid JSON: {e}")

        if not isinstance(data, list):
            raise CommandError("JSON root must be a list of objects.")

        self.stdout.write(f"\n{DIVIDER}")
        self.stdout.write(f"  Loading {len(data)} entries from {os.path.basename(path)}"
                          f"{'  [DRY RUN]' if dry_run else ''}")
        self.stdout.write(f"{DIVIDER}\n")

        errors   = []
        valid    = []

        for i, item in enumerate(data, 1):
            q = item.get("question", "").strip()
            if not q:
                errors.append(f"  [{i}] missing 'question'")
                continue

            # Accept both formats:
            # New:  expected_sources: [{document_name, madde}]
            # Old:  expected_maddes: [3, 5]  (defaults document_name to "KVKK")
            if "expected_sources" in item:
                sources = item["expected_sources"]
                for src in sources:
                    if "document_name" not in src or "madde" not in src:
                        errors.append(f"  [{i}] source missing document_name or madde: {src}")
                        break
            else:
                doc = item.get("document_name", "KVKK")
                sources = [
                    {"document_name": doc, "madde": m}
                    for m in item.get("expected_maddes", [])
                ]

            scenario = item.get("scenario", "user_validated")
            if scenario not in VALID_SCENARIOS:
                errors.append(f"  [{i}] unknown scenario '{scenario}'. Valid: {sorted(VALID_SCENARIOS)}")
                continue

            valid.append({
                "question":         q,
                "expected_answer":  item.get("expected_answer", "").strip(),
                "expected_sources": sources,
                "scenario":         scenario,
            })

        if errors:
            self.stdout.write(self.style.ERROR("  Validation errors:"))
            for e in errors:
                self.stdout.write(self.style.ERROR(e))
            self.stdout.write("")

        self.stdout.write(f"  Valid entries : {len(valid)}")
        self.stdout.write(f"  Invalid entries: {len(errors)}")

        if not valid:
            self.stdout.write("  Nothing to import.")
            return

        if dry_run:
            self.stdout.write("\n  [DRY RUN] — no changes made.")
            self.stdout.write(f"{DIVIDER}\n")
            return

        if reset:
            deleted, _ = GroundTruthEntry.objects.filter(promoted_from__isnull=True).delete()
            self.stdout.write(f"  Cleared {deleted} existing entries (promoted_from=None).")

        created   = 0
        updated   = 0
        skipped   = 0

        for entry in valid:
            obj, was_created = GroundTruthEntry.objects.update_or_create(
                question=entry["question"],
                defaults={
                    "expected_answer":  entry["expected_answer"],
                    "expected_sources": entry["expected_sources"],
                    "scenario":         entry["scenario"],
                    "promoted_from":    None,
                },
            )
            if was_created:
                created += 1
            else:
                updated += 1

        self.stdout.write(f"\n  Created : {created}")
        self.stdout.write(f"  Updated : {updated}")
        self.stdout.write(f"  Skipped : {skipped}")
        self.stdout.write(f"{DIVIDER}\n")
