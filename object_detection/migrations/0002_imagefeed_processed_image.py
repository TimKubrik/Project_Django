# Generated by Django 5.0.4 on 2024-04-21 13:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('object_detection', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='imagefeed',
            name='processed_image',
            field=models.ImageField(blank=True, null=True, upload_to='processed_images/'),
        ),
    ]
