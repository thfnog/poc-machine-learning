import os
import csv
import sys
import re

from surprise import Dataset
from surprise import Reader

from collections import defaultdict
import numpy as np

class AttractionsData:

    attractionID_to_name = {}
    name_to_attractionID = {}
    ratingsPath = 'datas/attraction_rating.csv'
    attractionPath = 'datas/attraction_info.csv'
    
    def loadAttractionsDataLatestSmall(self):

        # Look for files relative to the directory we are running from
        os.chdir(os.path.dirname(__file__))

        ratingsDataset = 0
        self.attractionID_to_name = {}
        self.name_to_attractionID = {}
        
        print("starting...")

        reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)

        ratingsDataset = Dataset.load_from_file(self.ratingsPath, reader=reader)

        with open(self.attractionPath, newline='', encoding='ISO-8859-1') as csvfile:
                attractionReader = csv.reader(csvfile)
                next(attractionReader)  #Skip header line
                for row in attractionReader:
                    attractionID = int(row[0])
                    attractionName = row[1]
                    self.attractionID_to_name[attractionID] = attractionName
                    self.name_to_attractionID[attractionName] = attractionID

        return ratingsDataset

    def getUserRatings(self, user):
        userRatings = []
        hitUser = False
        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                userID = int(row[0])
                if (user == userID):
                    attractionID = int(row[1])
                    rating = float(row[2])
                    userRatings.append((attractionID, rating))
                    hitUser = True
                if (hitUser and (user != userID)):
                    break

        return userRatings

    def getPopularityRanks(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                attractionID = int(row[1])
                ratings[attractionID] += 1
        rank = 1
        for attractionID, ratingCount in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[attractionID] = rank
            rank += 1
        return rankings
    
    def getGenres(self):
        genres = defaultdict(list)
        genreIDs = {}
        maxGenreID = 0
        with open(self.attractionPath, newline='', encoding='ISO-8859-1') as csvfile:
            movieReader = csv.reader(csvfile)
            next(movieReader)  #Skip header line
            for row in movieReader:
                attractionID = int(row[0])
                genreList = row[2].split('|')
                genreIDList = []
                for genre in genreList:
                    if genre in genreIDs:
                        genreID = genreIDs[genre]
                    else:
                        genreID = maxGenreID
                        genreIDs[genre] = genreID
                        maxGenreID += 1
                    genreIDList.append(genreID)
                genres[attractionID] = genreIDList
        # Convert integer-encoded genre lists to bitfields that we can treat as vectors
        for (attractionID, genreIDList) in genres.items():
            bitfield = [0] * maxGenreID
            for genreID in genreIDList:
                bitfield[genreID] = 1
            genres[attractionID] = bitfield            
        
        return genres
    
    def getYears(self):
        p = re.compile(r"(?:\((\d{4})\))?\s*$")
        years = defaultdict(int)
        with open(self.attractionPath, newline='', encoding='ISO-8859-1') as csvfile:
            movieReader = csv.reader(csvfile)
            next(movieReader)
            for row in movieReader:
                attractionID = int(row[0])
                title = row[1]
                m = p.search(title)
                year = m.group(1)
                if year:
                    years[attractionID] = int(year)
        return years
    
    def getMiseEnScene(self):
        mes = defaultdict(list)
        with open("LLVisualFeatures13K_Log.csv", newline='') as csvfile:
            mesReader = csv.reader(csvfile)
            next(mesReader)
            for row in mesReader:
                attractionID = int(row[0])
                avgShotLength = float(row[1])
                meanColorVariance = float(row[2])
                stddevColorVariance = float(row[3])
                meanMotion = float(row[4])
                stddevMotion = float(row[5])
                meanLightingKey = float(row[6])
                numShots = float(row[7])
                mes[attractionID] = [avgShotLength, meanColorVariance, stddevColorVariance,
                   meanMotion, stddevMotion, meanLightingKey, numShots]
        return mes
    
    def getAttractionName(self, attractionID):
        if attractionID in self.attractionID_to_name:
            return self.attractionID_to_name[attractionID]
        else:
            return ""
        
    def getAttractionID(self, attractionName):
        if attractionName in self.name_to_attractionID:
            return self.name_to_attractionID[attractionName]
        else:
            return 0