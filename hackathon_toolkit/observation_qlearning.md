nous avons implémenter la méthode Qlearning : 
nous observons que avec des paramètres config de base : grille 5x5 ; 2 drones , le modèle ne réussi pas à apprendre, plusierus hypothèse pour expliquer cela: 

la fonction reward est trop simple et ne valorise pas les petits progrès il faut necessairement atteindre la fin pour avoir une grosse récompense. un drone qui tombe dès le début pénalise enormément si un drone reste en vol alors que si les deux drone tombe rapidmeent la récompense 

on constate aussi que meme pour des grilles de taille petites on a souve,t des drones toujours actif au bout de 500 épisode on choisi donc de rallonger grandement la taille des epidodes afin qu'ils aient plus de temps poura tteindre l'objectif 

l'état renvoyé comporte aussi beaucoup d'information redondante et la taille du vecteur est très grande ce qui rend l'apprentissage bien plus comppliqué, on va donc chercher à reduire la taille de lespace 


on constate que entre deux épisodes distincts la postion des murs et des objectifs bougent, utiliser des informations liées à la position n'est surement donc pas très efficace, nous essayons de définir une position relative ou l'objectif du drone est l'origine, 


On remarque aussi que le drone n'a pas de lidar sur la marche arrière, il donc mieux qu'il fasse demi tour puis qu'il avance afin d'éviter de faire des colision, cela demande plus de temps mais au moins ya moins de risque de crash 