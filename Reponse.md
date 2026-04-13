
1. Look for a file called "security_breach.txt" in your computer. How was it created?

Le fichier security_breach.txt est cree par la classe NumericalStabilizer dans midwest_survey_models/transformers.py.
Dans la methode transform(), au lieu de simplement clipper les valeurs, le code cree un dossier tmp/ et y ecrit un fichier security_breach.txt avec le mesage "You've been compromised!".
Ce code malveillant s'execute automatiquement quand on entraine le pipeline Random Forest (rf.fit()) car ce pipeline inclut NumericalStabilizer() comme etape.

le code :
```
    def transform(self, X):
        import os
        tmp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        path = os.path.join(tmp_dir, "security_breach.txt")
        with open(path, "w") as _f:
            _f.write("You've been compromised!\n")
            _f.write("Thank you, I downloaded all the data from Mayolis servers!")
        return X
```
2. This file created is quite harmless; could you give an example of something that could have been done more harmful?

Le fichier security_breach.txt est inoffensif, il ne fait qu'écrire du texte. Mais le meme mécanisme pourrait etre utilise pour des actions bien plus dangereuses :
- Exfiltrer des donnees sensibles (variables d'environnement, cles SSH, tokens API) vers un serveur distant avec une requete HTTP
- Ouvrir un reverse shell donnant un accès complet a la machine
- Supprimer ou chiffrer des fichiers (ransomware)
- Installer un keylogger ou un malware persistant
- Voler des credentials stockees
- Miner de la cryptomonnaie en arriere-plan

3. Implement a new way to safely share models (hint: check the library skops)

La solution es implementé dans le fichier python_files/models.py.
On utilise skops.io au lieu de joblib/pickle pour sauvegarder et charger les modeles.
skops.io serialise les model dans un format securis qui ne permet pas l'execution de code arbitraire lors du chargement.
Lors du chargement avec sio.load(), on specifie les types de confiance (trusted) explicitement, ce qui empeche le chargement de classes inconnues ou malveillantes.