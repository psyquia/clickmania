from rest_framework import serializers
from .models import *

class CollectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Collection
        fields = ('collectionID', 'name', 'display_name', 'description', 'img_url')

class ArtSerializer(serializers.ModelSerializer):
    img_url = serializers.ReadOnlyField()
    thumb_url = serializers.ReadOnlyField()
    class Meta:
        model = Art
        fields = ('artID', 'title', 'filename', 'rarity', 'collection', 'img_url', 'thumb_url')

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'password', 'coins', 'art')
        extra_kwargs = {
            'password': {'write_only': True}
        }
    
    def create(self, validated_data):
        password = validated_data.pop('password', None)
        instance = self.Meta.model(**validated_data)
        if password is not None:
            instance.set_password(password)
        instance.save()
        return instance

class OwnSerializer(serializers.ModelSerializer):
    duplicates = serializers.ReadOnlyField()
    class Meta:
        model = Own
        fields = ('ownID', 'user', 'art', 'duplicates')

class SaleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Sale
        fields = ('saleID', 'seller', 'buyer', 'ownership', 'art', 'price', 'available', 'sold', 'postDate', 'purchaseDate')