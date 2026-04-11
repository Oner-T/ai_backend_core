from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('intelligence', '0003_alter_documentchunk_embedding'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='documentsection',
            name='name',
        ),
        migrations.AddField(
            model_name='documentsection',
            name='number',
            field=models.IntegerField(default=0, help_text='e.g., 1 for BİRİNCİ BÖLÜM', unique=True),
            preserve_default=False,
        ),
    ]
