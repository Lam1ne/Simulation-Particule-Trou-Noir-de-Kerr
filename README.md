Simulateur de Géodésiques autour d'un Trou Noir de Kerr
Bienvenue dans le simulateur de trajectoires autour d'un trou noir de Kerr. Ce projet est une application interactive en C++ qui permet de simuler et de visualiser les géodésiques d'une particule autour d'un trou noir en rotation, en utilisant l'intégration numérique de Runge-Kutta d'ordre 4.
Table des matières
•	#fonctionnalités
•	#pré-requis
•	#utilisation
•	#aperçu
---
Fonctionnalités
•	Simulation des géodésiques : Calcule les trajectoires de particules autour d'un trou noir de Kerr en tenant compte des effets relativistes.
•	Visualisation 3D interactive : Permet de visualiser en temps réel la trajectoire, l'horizon des événements, l'ergosphère et le disque d'accrétion.
•	Contrôles en temps réel :
        • Ajustez les constantes de mouvement (énergie, moment angulaire, etc.).
        • Modifiez les conditions initiales de la particule.
•	Contrôlez les paramètres du trou noir (masse, spin).
•	Contrôlez la caméra pour explorer la simulation sous différents angles.
•	Interface utilisateur avec ImGui : Interface intuitive pour modifier les paramètres et contrôler la simulation.
•	Exportation des données : Exportez les données de trajectoire vers un fichier CSV pour une analyse ultérieure.
---
Pré-requis
Assurez-vous d'avoir les éléments suivants installés sur votre système :
•	C++14 ou une version supérieure du compilateur C++.
•	GLFW : Bibliothèque pour la gestion des fenêtres et des entrées.
•	GLAD : Chargeur d'extensions OpenGL.
•	OpenGL : API pour le rendu graphique 2D et 3D.
•	GLM : Bibliothèque mathématique pour OpenGL.
•	ImGui : Interface graphique immédiate pour C++.
•	stb_image : Bibliothèque pour le chargement d'images.
•	CMake (optionnel) : Pour faciliter la compilation multiplateforme.
