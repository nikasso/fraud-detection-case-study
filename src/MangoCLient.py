from pymongo import MongoClient
import pandas as pd
class FraudMongo(object):
   def __init__(self):
        self.client=MongoClient()

   def insertOneToMongoDb(self,df):
        self.client.dbs.collec.insert_many(df.to_dict('records'))
   def getCollectionAsDataframe(self):
        df=pd.DataFrame(list(self.client.dbs.collec.find()))
        del df['_id']
        return df
