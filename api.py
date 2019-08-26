from flask import Flask
from flask_restful import Resource, Api, reqparse
import numpy as np
import tensorflow as tf
from RelationExtractor import RelationExtractor
# from Summary import Summary


app = Flask(__name__)
api = Api(app)

# class for getting result from model
relExtractor = RelationExtractor()


class GetData(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('corpus', type=str)
            args = parser.parse_args()

            _corpus = args['corpus']
            relation_list = relExtractor.get_result(_corpus)
            summary = "this is summary!"
            return {'relation': relation_list, 'summary': summary}
        except Exception as e:
            return {'error': str(e)}

api.add_resource(GetData, '/result')

if __name__ == '__main__':
    app.run(debug=True)