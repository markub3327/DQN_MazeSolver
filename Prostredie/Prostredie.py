import numpy as np
import os
import random

from Prostredie.EnvItem import *

class Prostredie:
    # Startovacie body
    startPositions = [(0,1), (9,1), (0,5)]
    
    # body v bludisku 
    prostredie = []
    
    def __init__(self, area_width, area_height, prostredie=None):
        self.area_width = area_width
        self.area_height = area_height
        self.currentPositionX = None
        self.currentPositionY = None
        
        if (prostredie != None):
            for i in range(area_height * area_width):
                if (prostredie[i] == Jablko.Tag):
                    self.prostredie.append(Jablko())
                elif (prostredie[i] == Mina.Tag):
                    self.prostredie.append(Mina())
                elif (prostredie[i] == Cesta.Tag):
                    self.prostredie.append(Cesta())
                elif (prostredie[i] == Priepast.Tag):
                    self.prostredie.append(Priepast())
                elif (prostredie[i] == Vychod.Tag):
                    self.prostredie.append(Vychod())
        
    def GenerateItem(self, item):
            x = np.random.randint(1, self.area_width-1)
            y = np.random.randint(1, self.area_height-1)

            while (self.getId(x, y) != Cesta.Tag):
                x = np.random.randint(1, self.area_width-1)
                y = np.random.randint(1, self.area_height-1)

            # prepis novou polozkou sveta
            self.prostredie[y*self.area_width + x] = item

    def Hodnotenie(self, x, y):
            if (self.getId(x, y) == Priepast.Tag):
                return -1.00    # Smrt
            elif (self.getId(x, y) == Jablko.Tag):
                return +0.10    # Jablcko (odmena)
            elif (self.getId(x, y) == Mina.Tag):
                return -0.15    # Mina (trest)
            elif (self.getId(x, y) == Vychod.Tag):
                return +1.00    # Dalsi level      
            else:
                return -0.04    # Najkratsia cesta k vychodu

    def NahradObjekty(self, tag, item):
        for i in range(self.area_height * self.area_width):
            if (self.prostredie[i].id == tag):
                self.prostredie[i] = item
        
    def render(self):
            os.system('clear')

            for i in range(self.area_width * 4 + 1):
                print('-', end='')
            print()

            for y in range(self.area_height):
                for x in range(self.area_width):
                    if (self.currentPositionX == x and self.currentPositionY == y):
                        # Vykresli agenta
                        print("|\033[0;103;30m A \033[0m", end='')
                    else:                       
                        print(f"|{self.prostredie[y*self.area_width + x]}", end='')
                   
                print("|")
                for i in range(self.area_width * 4 + 1):
                    print('-', end='')
                print()

    def reset(self):  
        pos = random.choice(self.startPositions)
        self.currentPositionX = pos[0]
        self.currentPositionY = pos[1]
        
        # Vymaz vsetky jablka a miny
        self.NahradObjekty(Jablko.Tag, Cesta())
        self.NahradObjekty(Mina.Tag, Cesta())

        for i in range(random.randint(1,5)):
            self.GenerateItem(Jablko())
        
        for i in range(random.randint(1,5)):
            self.GenerateItem(Mina())

        state = [
            self.currentPositionX/10.0,
            self.currentPositionY/10.0
        ]
        state.extend(self.Radar())

        return np.array(state)

    def sample(self):
        akcia = np.random.randint(0, 4)
        return akcia

    def step(self, action):
        oldX = self.currentPositionX
        oldY = self.currentPositionY
        #print(self.currentPositionX, self.currentPositionY)

        if action == 0:
            self.currentPositionY += 1
        elif action == 1:
            self.currentPositionY -= 1
        elif action == 2:
            self.currentPositionX -= 1
        elif action == 3:
            self.currentPositionX += 1

        #print(action, self.currentPositionX, self.currentPositionY)

        # skontroluj spravnost kroku v hre
        if (self.JeMimoAreny(self.currentPositionX, self.currentPositionY, 10, 10)):
            self.currentPositionX = oldX
            self.currentPositionY = oldY                
       
            reward = -50.0
            done = True   # koniec hry
        else:
            reward = self.Hodnotenie(self.currentPositionX, self.currentPositionY)

            # hra konci najdenim ciela
            done = True if self.getId(self.currentPositionX, self.currentPositionY) == Vychod.Tag else False

        state = [
            self.currentPositionX/10.0,
            self.currentPositionY/10.0
        ]
        state.extend(self.Radar())


        # Agent zobral jablko
        if (self.getId(self.currentPositionX, self.currentPositionY) == Jablko.Tag):
            self.prostredie[self.currentPositionY*self.area_width + self.currentPositionX] = Cesta()
                                                
        # Agent aktivoval minu
        if (self.getId(self.currentPositionX, self.currentPositionY) == Mina.Tag):
            self.prostredie[self.currentPositionY*self.area_width + self.currentPositionX] = Cesta()

        return (np.array(state), reward, done)
    
    def getId(self, x, y):
        return self.prostredie[y*self.area_width + x].id

    def upravaDatRadaru(self, x, y):
        if (self.getId(x, y) == Priepast.Tag):
            return [0, 0, 0]
        elif (self.getId(x, y) == Jablko.Tag):
            return [0, 1, 0]
        elif (self.getId(x, y) == Mina.Tag):
            return [0, 1, 1]
        elif (self.getId(x, y) == Cesta.Tag):
            return [0, 0, 1]
        elif (self.getId(x, y) == Vychod.Tag):
            return [1, 0, 0]

    def Radar(self):
        scan = []

        # Hore
        if self.JeMimoAreny(self.currentPositionX, self.currentPositionY-1, 10, 10) == False:
            scan.extend(self.upravaDatRadaru(self.currentPositionX, self.currentPositionY-1))
        else:
            scan.extend([0, 0, 0])

        # Vpravo-Hore
        if self.JeMimoAreny(self.currentPositionX+1, self.currentPositionY-1, 10, 10) == False:
            scan.extend(self.upravaDatRadaru(self.currentPositionX+1, self.currentPositionY-1))
        else:
            scan.extend([0, 0, 0])

        # Vpravo
        if self.JeMimoAreny(self.currentPositionX+1, self.currentPositionY, 10, 10) == False:
            scan.extend(self.upravaDatRadaru(self.currentPositionX+1, self.currentPositionY))
        else:
            scan.extend([0, 0, 0])

        # Vpravo-Dole
        if self.JeMimoAreny(self.currentPositionX+1, self.currentPositionY+1, 10, 10) == False:
            scan.extend(self.upravaDatRadaru(self.currentPositionX+1, self.currentPositionY+1))
        else:
            scan.extend([0, 0, 0])

        # Dole
        if self.JeMimoAreny(self.currentPositionX, self.currentPositionY+1, 10, 10) == False:
            scan.extend(self.upravaDatRadaru(self.currentPositionX, self.currentPositionY+1))
        else:
            scan.extend([0, 0, 0])

        # Vlavo-Dole
        if self.JeMimoAreny(self.currentPositionX-1, self.currentPositionY+1, 10, 10) == False:
            scan.extend(self.upravaDatRadaru(self.currentPositionX-1, self.currentPositionY+1))
        else:
            scan.extend([0, 0, 0])

        # Vlavo
        if self.JeMimoAreny(self.currentPositionX-1, self.currentPositionY, 10, 10) == False:
            scan.extend(self.upravaDatRadaru(self.currentPositionX-1, self.currentPositionY))
        else:
            scan.extend([0, 0, 0])

        # Vlavo-Hore
        if self.JeMimoAreny(self.currentPositionX-1, self.currentPositionY-1, 10, 10) == False:
            scan.extend(self.upravaDatRadaru(self.currentPositionX-1, self.currentPositionY-1))
        else:
            scan.extend([0, 0, 0])

        return scan

    def JeMimoAreny(self, x, y, sirka, vyska):
        if (x >= 0 and x < sirka and y >= 0 and y < vyska):
            return False

        return True
