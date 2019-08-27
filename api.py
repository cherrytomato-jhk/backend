from flask import Flask
from flask_restful import Resource, Api, reqparse
import numpy as np
import tensorflow as tf
from RelationExtractor import RelationExtractor
from Summary import Summarization
from flask_cors import CORS,cross_origin

app = Flask(__name__)

CORS(app)

api = Api(app)


# class for getting result from model
relGraph = tf.Graph()
suGraph = tf.Graph()
with relGraph.as_default():
    relExtractor = RelationExtractor()
with suGraph.as_default():
    summarization = Summarization()


class GetData(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('corpus', type=str)
            args = parser.parse_args()

            _corpus = args['corpus']
            try:
                relation_list = relExtractor.get_result(_corpus)
            except Exception as e:
                print("[Rel Error]", e)
            try:
                summary = summarization.get_result(_corpus)
            except Exception as e:
                print("[Summary Error]", e)
            return {'relation': relation_list, 'summary': summary}
        except Exception as e:
            return {'error': str(e)}

api.add_resource(GetData, '/result')

if __name__ == '__main__':
    app.run(debug=True)