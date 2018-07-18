import tweepy

Consumer_Key='YxIb1ALNg9xJ514nzHeS0Qh9M'
Consumer_Secret='QbrgGOXgaqdOL1ln592PTtTCIxpO7Qmzt6ElB7Jwu6XJG7YLjZ'
Access_Token='2181668512-iGcL8j4hURiLYdAcfvSkUpwsf498DVIgrHbXp0c'
Access_Token_Secret='PwbuRMSENW40ZJAmwtlFeabUUUS9varYF2jD1xDeXF8l3'

auth=tweepy.OAuthHandler(Consumer_Key,Consumer_Secret)
auth.set_access_token(Access_Token,Access_Token_Secret)
api=tweepy.API(auth)
'''for i in tweepy.Cursor(api.home_timeline).items(5):
    print(i.text)'''
from tweepy import Stream
from tweepy.streaming import StreamListener
import json
class MyListener(StreamListener):

    def on_data(self, data):
        try:
            with open('python1.json', 'a') as f:
                json.dump(data,f)
                return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True

    def on_error(self, status):
        print(status)
        return True

twitter_stream = Stream(auth, MyListener())
twitter_stream.filter(track=['#PUBG_MOBILE'])
