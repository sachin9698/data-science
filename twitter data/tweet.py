import tweepy

Consumer_Key='YxIb1ALNg9xJ514nzHeS0Qh9M'
Consumer_Secret='QbrgGOXgaqdOL1ln592PTtTCIxpO7Qmzt6ElB7Jwu6XJG7YLjZ'
Access_Token='2181668512-iGcL8j4hURiLYdAcfvSkUpwsf498DVIgrHbXp0c'
Access_Token_Secret='PwbuRMSENW40ZJAmwtlFeabUUUS9varYF2jD1xDeXF8l3'

auth=tweepy.OAuthHandler(Consumer_Key,Consumer_Secret)
auth.set_access_token(Access_Token,Access_Token_Secret)
api=tweepy.API(auth)

# for i in tweepy.Cursor(api.friends).items():
#     print(i)

# for status in tweepy.Cursor(api.home_timeline).items(10):
#     # Process a single status
#     print(status.text)

for status in tweepy.Cursor(api.home_timeline).items(10):
    # Process a single status
    process_or_store(status._json)
