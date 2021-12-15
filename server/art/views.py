from django.shortcuts import render
from .serializers import *
from rest_framework import viewsets, permissions, generics, filters
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.exceptions import AuthenticationFailed
from django_filters.rest_framework import DjangoFilterBackend
from django.utils import timezone
from .models import *
from rest_framework import exceptions
import pytz
import jwt
import numpy as np
import environ
# Initialise environment variables
env = environ.Env()
environ.Env.read_env()

class CollectionView(viewsets.ModelViewSet):
    permission_classes = [
        permissions.DjangoModelPermissionsOrAnonReadOnly
    ]
    serializer_class = CollectionSerializer
    queryset = Collection.objects.all()

class ArtView(viewsets.ModelViewSet):
    permission_classes = [
        permissions.DjangoModelPermissionsOrAnonReadOnly
    ]
    serializer_class = ArtSerializer
    queryset = Art.objects.all()


class OwnView(viewsets.ModelViewSet):
    queryset = Own.objects.all()
    serializer_class = OwnSerializer
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter,]
    filterset_fields = ['user', 'art']

class SaleView(viewsets.ModelViewSet):
    serializer_class = SaleSerializer
    queryset = Sale.objects.all()

class RegisterView(APIView):
    permission_classes = [
        permissions.AllowAny
    ]
    def post(self, request):
        serializer = UserSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)

class LoginView(APIView):
    permission_classes = [
        permissions.AllowAny
    ]
    def post(self, request):
        username = request.data['username']
        password = request.data['password']
        user = User.objects.filter(username=username).first()
        if user is None:
            raise AuthenticationFailed('User not found!')
        if not user.check_password(password):
            raise AuthenticationFailed('Incorrect password!')
        payload = {
            'id': user.id,
        }
        token = jwt.encode(payload, env('JWT_KEY'), algorithm='HS256')
        serializer = UserSerializer(user)
        return Response({'user': serializer.data, 'jwt': token})

class GetSaleListingsView(APIView):
    def get(self, request):
        sales = Sale.objects.filter(available=True).filter(sold=False)

        sale_listings = []
        for sale in sales:
            sale_serializer = SaleSerializer(sale)
            art_serializer = ArtSerializer(sale.art)
            user_serializer = UserSerializer(sale.seller)
            collection = Collection.objects.filter(collectionID=art_serializer.data['collection']).first()
            collection_serializer = CollectionSerializer(collection)

            sale_listings.append({
                'sale': sale_serializer.data,
                'art': art_serializer.data,
                'collection': collection_serializer.data,
                'seller': user_serializer.data,
            })

        return Response(sale_listings)

class GetUserView(APIView):
    def get(self, request):
        authorization_header = request.headers.get('Authorization')

        if not authorization_header:
            return None
        access_token = authorization_header.split(' ')[1]
        try:
            payload = jwt.decode(
                access_token, env('JWT_KEY'), algorithms=['HS256'])

        except jwt.ExpiredSignatureError:
            raise exceptions.AuthenticationFailed('access_token expired')
        except IndexError:
            raise exceptions.AuthenticationFailed('Token prefix missing')
        user = User.objects.filter(id=payload['id']).first()
        serializer = UserSerializer(user)
        return Response(serializer.data)

class UpdateUserView(APIView):
    def post(self, request):
        print(request.data)
        user = User.objects.filter(username=request.data['username']).first()
        if user is None:
            raise AuthenticationFailed('User not found!')
        user.coins = request.data['coins']
        user.email = request.data['email']
        user.save()
        serializer = UserSerializer(user)
        return Response(serializer.data)

class BuyLootBoxView(APIView):
    def post(self, request):
        user = User.objects.filter(username=request.data['user']['username']).first()
        if user is None:
            raise AuthenticationFailed('User not found!')
        collection = Collection.objects.filter(collectionID=request.data['collection']['collectionID']).first()
        if collection is None:
            raise AuthenticationFailed('Collection does not exist!')

        ### Check balance and remove cost from player balance
        if user.coins < 100:
            return Response("Insufficient coins!")
        user.coins -= 100
        user.save()
        user_serializer = UserSerializer(user)

        ### Lootbox RNG logic ###
        # roll the rarity (1 = worst, 5 = best[rarest])
        rarityRoll = np.random.choice(np.arange(1, 6), p=[0.33, 0.31, 0.25, 0.1, 0.01]);
        # filter art by collection and rarity
        possibleArt = Art.objects.filter(collection=collection.collectionID).filter(rarity=rarityRoll)
        # roll for the specific piece of art
        idxRoll = np.random.randint(0, len(possibleArt))
        chosenArt = possibleArt[idxRoll]
        art_serializer = ArtSerializer(chosenArt)

        ### Add owns relationship ###
        ownObj = Own(user=user, art=chosenArt)
        ownObj.save()
        print("new own obj:", ownObj)

        return Response({
            'art': art_serializer.data,
            'user': user_serializer.data
        })

class GetUserArtsView(APIView):
    def post(self, request):
        user = User.objects.filter(id=request.data['id']).first()
        ownedArt = []

        ownsList = Own.objects.filter(user=user)
        for own in ownsList:
            art = Art.objects.filter(artID=own.art.artID).first()
            #check if art on sale
            existingSales = Sale.objects.filter(ownership=own, available=True)
            isOnSale = len(existingSales) > 0
            data = {
                'own': OwnSerializer(own).data,
                'art': ArtSerializer(art).data,
                'isOnSale': isOnSale,
                'collection_name': art.collection.display_name
            }
            ownedArt.append(data)

        return Response(ownedArt)

class CreateSaleView(APIView):
    def post(self, request):
        own_id = request.data['ownID']
        price = request.data['price']
        #check if price negative
        if price < 0:
            raise AuthenticationFailed('Price cannot be negative')
        own = Own.objects.filter(ownID=own_id).first()
        # check if current art already on sale
        existingSales = Sale.objects.filter(ownership=own, available=True)
        if len(existingSales) > 0:
                raise AuthenticationFailed('This is already for sale on the market')
        SaleObj = Sale(seller=own.user, art=own.art, price=price, ownership=own)
        SaleObj.save()
        return Response({
            'success': True
        })

class StopSellingView(APIView):
    def post(self, request):
        Sale.objects.filter(saleID=request.data['saleID']).delete()
        return Response({
            'success': True
        })

class BuyFromMarketView(APIView):
    def post(self, request):
        buyer = User.objects.get(pk=request.data['buyerID'])
        sale = Sale.objects.get(pk=request.data['saleID'])
        seller = User.objects.get(pk=sale.seller.id)
        own = Own.objects.get(pk=sale.ownership.ownID)
        if sale.sold:
            raise AuthenticationFailed('Sorry, this art has already been sold.')
        if not sale.available:
            raise AuthenticationFailed('Sorry, this art is no longer available on the market.')
        if sale.price > buyer.coins:
            raise AuthenticationFailed('Sorry, you do not have enough coins.')
        sale.buyer = buyer
        sale.sold = True
        sale.available = False
        sale.purchaseDate = timezone.now()
        sale.save()
        buyer.coins = buyer.coins - sale.price
        own.user = buyer
        seller.coins = seller.coins + sale.price
        own.save()
        buyer.save()
        seller.save()
        user_serializer = UserSerializer(buyer)
        return Response({
            'success': True,
            'user': user_serializer.data
        })
        
        
        
        


