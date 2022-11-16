

import numpy as np
import cv2
import argparse
import imutils






#import de l'image à analyser
imageAnalysee = cv2.imread('desert_H_angle.jpg')
imAffich5 = cv2.resize(imageAnalysee, (500, 375))
cv2.imshow("image à analyser", imAffich5)
cv2.waitKey(0)
cv2.destroyAllWindows()
imageFinale = imageAnalysee.copy()

#passage en HSV
hsvFrame = cv2.cvtColor(imageAnalysee, cv2.COLOR_BGR2HSV)

#création du masque de détection du vert
green_lower = np.array([25, 52, 72], np.uint8)
green_upper = np.array([102, 255, 255], np.uint8)
green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

kernal = np.ones((5, 5), "uint8")

green_mask = cv2.dilate(green_mask, kernal)


#trouver les éléments verts dans l'image
contours, hierarchy = cv2.findContours(green_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)


#analyse des éléments verts et indication de ceux -ci sur l'image
i=0
imageAnalysee2 = imageAnalysee.copy()
for pic, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if(area > 300):
        i=i+1
        x1, y1, w1, h1 = cv2.boundingRect(contour)
        imageAnalysee2 = cv2.rectangle(imageAnalysee2, (x1, y1),
                                       (x1 + w1, y1 + h1),
                                       (0, 255, 0), 2)

        cropped = imageAnalysee2[y1:y1+h1, x1:x1+w1] #isolation de la zone verte sur une image

        if i>0:
            #affichage de l'image (zone indiquée sur l'image de départ) sur l'écran (réduction de la taille pour image en entier sur l'écran)
            imAffich = cv2.resize(imageAnalysee2, (500, 375))
            """cv2.imshow("zone dans le desert", imAffich)
            cv2.waitKey(0)
            cv2.destroyAllWindows()"""

            #affichage de la zone isolée
            imAffich2 = cv2.resize(cropped, (300, 300))
            """cv2.imshow("zone isolée", imAffich2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()"""

            """enlever l'arrière plan vert"""
            #conversion de l'image isolée en HSV
            img = imAffich2.copy()
            img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # lower mask (0-10) masque de rouge bas
            lower_red = np.array([0,50,50])
            upper_red = np.array([10,255,255])
            mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

            # upper mask (170-180) masque de rouge haut
            lower_red = np.array([170,50,50])
            upper_red = np.array([180,255,255])
            mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

            # join my masks
            mask = mask0+mask1

            # set my output img to zero everywhere except my mask
            output_img = img.copy()
            output_img[np.where(mask==0)] = 255


            #affichage de la zone sans fond vert
            """cv2.imshow("H de la zone sur fond blanc", output_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()"""


            #conversion de l'image en niveaux de gris et affichage
            color_test_image = output_img.copy()
            gray_test_image = cv2.cvtColor(color_test_image, cv2.COLOR_BGR2GRAY)
            """cv2.imshow('H de la zone sur fond blanc en noir et blanc', gray_test_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()"""

            #inversion des niveaux de gris
            ret, thresh = cv2.threshold(gray_test_image, 170, 255, cv2.THRESH_BINARY_INV)
            """cv2.imshow("H de la zone détectée en niveaux de gris inversés", thresh)
            cv2.waitKey(0)
            cv2.destroyAllWindows()"""



            #recherche du contour du H
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                # on vérifie la taille du contour pour éviter de traiter un 'défaut'
                if cv2.contourArea(cnt) > 50:













                    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,  #trouve les contours du H
                                        cv2.CHAIN_APPROX_SIMPLE)
                    cnts = imutils.grab_contours(cnts)
                    c = max(cnts, key=cv2.contourArea)
                    epsilon = 0.01*cv2.arcLength(c,True)            #epsilon va définir la précision de l'approximation du contour
                    approx = cv2.approxPolyDP(c,epsilon,True)       #polygonal en fonction de la longueur du contour trouvé
                    output = output_img.copy()
                    cv2.drawContours(output, approx, -1, (0, 255, 0), 3)
                    (x, y, w, h) = cv2.boundingRect(c)
                    """
                    text = "original, num_pts={}".format(len(approx))
                    print(text)
                    cv2.putText(output, text, (x-50, y - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)
                    cv2.imshow("Original Contour", output)
                    cv2.waitKey(0)
                    """
                    if(len(approx)==12):                    #si le nombre de sommets/d'arêtes du polygone est 12
                        imageFinale = cv2.rectangle(imageFinale, (x1, y1),    #on trace un rectangle bleu sur la zone détectée
                                        (x1 + w1, y1 + h1),
                                            (255, 0, 0), -1)

cv2.imshow("zone d'atterrissage", cv2.resize(imageFinale, (500, 375)))
cv2.waitKey(0)
cv2.destroyAllWindows()








