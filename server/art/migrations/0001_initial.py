# Generated by Django 3.2.8 on 2021-12-07 04:41

from django.conf import settings
import django.contrib.auth.models
import django.contrib.auth.validators
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('auth', '0012_alter_user_first_name_max_length'),
    ]

    operations = [
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('password', models.CharField(max_length=128, verbose_name='password')),
                ('last_login', models.DateTimeField(blank=True, null=True, verbose_name='last login')),
                ('is_superuser', models.BooleanField(default=False, help_text='Designates that this user has all permissions without explicitly assigning them.', verbose_name='superuser status')),
                ('username', models.CharField(error_messages={'unique': 'A user with that username already exists.'}, help_text='Required. 150 characters or fewer. Letters, digits and @/./+/-/_ only.', max_length=150, unique=True, validators=[django.contrib.auth.validators.UnicodeUsernameValidator()], verbose_name='username')),
                ('first_name', models.CharField(blank=True, max_length=150, verbose_name='first name')),
                ('last_name', models.CharField(blank=True, max_length=150, verbose_name='last name')),
                ('is_staff', models.BooleanField(default=False, help_text='Designates whether the user can log into this admin site.', verbose_name='staff status')),
                ('is_active', models.BooleanField(default=True, help_text='Designates whether this user should be treated as active. Unselect this instead of deleting accounts.', verbose_name='active')),
                ('date_joined', models.DateTimeField(default=django.utils.timezone.now, verbose_name='date joined')),
                ('email', models.EmailField(max_length=254, unique=True, verbose_name='user email address')),
                ('coins', models.IntegerField(default=0, verbose_name='amount of currency user owns')),
            ],
            options={
                'verbose_name': 'user',
                'verbose_name_plural': 'users',
                'abstract': False,
            },
            managers=[
                ('objects', django.contrib.auth.models.UserManager()),
            ],
        ),
        migrations.CreateModel(
            name='Art',
            fields=[
                ('artID', models.BigAutoField(primary_key=True, serialize=False, verbose_name='art ID')),
                ('title', models.CharField(max_length=255, unique=True, verbose_name='title of artwork')),
                ('filename', models.CharField(max_length=255, verbose_name='filename')),
                ('rarity', models.IntegerField(default=1, verbose_name='rarity level of artwork')),
            ],
        ),
        migrations.CreateModel(
            name='Collection',
            fields=[
                ('collectionID', models.BigAutoField(primary_key=True, serialize=False, verbose_name='collection ID')),
                ('name', models.CharField(max_length=255, unique=True, verbose_name='name of collection')),
                ('display_name', models.CharField(default='', max_length=255, verbose_name='display name of collection')),
                ('description', models.CharField(default='', max_length=500, verbose_name='description of the collection')),
            ],
        ),
        migrations.CreateModel(
            name='Own',
            fields=[
                ('ownID', models.BigAutoField(primary_key=True, serialize=False, verbose_name='ownership instance ID')),
                ('art', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='art.art', verbose_name='ID of art')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL, verbose_name='ID of art owner')),
            ],
        ),
        migrations.CreateModel(
            name='Sale',
            fields=[
                ('saleID', models.BigAutoField(primary_key=True, serialize=False, verbose_name='sale ID')),
                ('price', models.IntegerField(default=10, verbose_name='amount of currency seller requests')),
                ('available', models.BooleanField(default=True, verbose_name='is the product available for sale')),
                ('sold', models.BooleanField(default=False, verbose_name='has the product been sold')),
                ('postDate', models.DateTimeField(auto_now_add=True)),
                ('purchaseDate', models.DateTimeField(blank=True, null=True)),
                ('art', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='art.art', verbose_name='ID of art')),
                ('buyer', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='buyer', to=settings.AUTH_USER_MODEL, verbose_name='ID of art buyer')),
                ('ownership', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='art.own', verbose_name='ID of ownership instance')),
                ('seller', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='seller', to=settings.AUTH_USER_MODEL, verbose_name='ID of art seller')),
            ],
        ),
        migrations.AddField(
            model_name='art',
            name='collection',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='art.collection', verbose_name='ID of collection art belongs to'),
        ),
        migrations.AddField(
            model_name='user',
            name='art',
            field=models.ManyToManyField(through='art.Own', to='art.Art', verbose_name='art owned by user'),
        ),
        migrations.AddField(
            model_name='user',
            name='groups',
            field=models.ManyToManyField(blank=True, help_text='The groups this user belongs to. A user will get all permissions granted to each of their groups.', related_name='user_set', related_query_name='user', to='auth.Group', verbose_name='groups'),
        ),
        migrations.AddField(
            model_name='user',
            name='user_permissions',
            field=models.ManyToManyField(blank=True, help_text='Specific permissions for this user.', related_name='user_set', related_query_name='user', to='auth.Permission', verbose_name='user permissions'),
        ),
    ]
