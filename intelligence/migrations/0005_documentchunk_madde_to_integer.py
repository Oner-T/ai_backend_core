from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("intelligence", "0004_documentsection_name_to_number"),
    ]

    operations = [
        # Step 1: extract the number from "MADDE X" in-place, then cast to integer
        migrations.RunSQL(
            sql="""
                ALTER TABLE intelligence_documentchunk
                ALTER COLUMN madde TYPE integer
                USING (regexp_match(madde, '\\d{1,3}'))[1]::integer;
            """,
            reverse_sql="""
                ALTER TABLE intelligence_documentchunk
                ALTER COLUMN madde TYPE varchar(255)
                USING 'MADDE ' || madde::text;
            """,
        ),
        # Step 2: let Django know about the new field type
        migrations.AlterField(
            model_name="documentchunk",
            name="madde",
            field=models.IntegerField(blank=True, null=True),
        ),
    ]
