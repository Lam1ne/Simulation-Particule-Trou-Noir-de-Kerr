// main.cpp
#include "imgui/imgui.h"
#include "imgui/imgui_impl_opengl3.h"
#include "imgui/imgui_impl_glfw.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stb_image.h>

// Forcer l'utilisation des radians 
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

using namespace std;

const double PI = 3.14159265358979323846;

// Structure pour les conditions initiales
struct ConditionsInitiales {
    double r;
    double theta;
    double phi;
    double tau;
    double p_r;
    double p_theta;
};

// Structure pour l'état des géodésiques
struct EtatGeodesique {
    double r;
    double theta;
    double phi;
    double tau;
    double p_r;
    double p_theta;

    // Constructeur de conversion depuis ConditionsInitiales
    EtatGeodesique(const ConditionsInitiales& init) {
        r = init.r;
        theta = init.theta;
        phi = init.phi;
        tau = init.tau;
        p_r = init.p_r;
        p_theta = init.p_theta;
    }

    // Constructeur par défaut
    EtatGeodesique() : r(0), theta(0), phi(0), tau(0), p_r(0), p_theta(0) {}
};

// Conversion des coordonnées sphériques en cartésiennes
tuple<double, double, double> cartesian(double r, double theta, double phi, double a) {
    double x = sqrt(r * r + a * a) * sin(theta) * cos(phi);
    double y = sqrt(r * r + a * a) * sin(theta) * sin(phi);
    double z = r * cos(theta);
    return make_tuple(x, y, z);
}

// Méthode de Runge-Kutta 4 pour l'intégration des géodésiques
EtatGeodesique rungeKutta4(const EtatGeodesique& etat, double dt, double a, double mu, double E, double L, double Q, double M) {
    auto derivatives = [&](const EtatGeodesique& s) -> EtatGeodesique {
        double r = s.r;
        double theta = s.theta;
        double p_r = s.p_r;
        double p_theta = s.p_theta;
        double Sigma = r * r + a * a * cos(theta) * cos(theta);
        double Delta = r * r - 2.0 * M * r + a * a;
        double k = Q + L * L + a * a * (E * E + mu);

        // Éviter la division par zéro
        if (Sigma < 1e-6) Sigma = 1e-6;
        if (Delta < 1e-6) Delta = 1e-6;

        // Calcul des dérivées
        double dr_dtau = Delta / Sigma * p_r;
        double dtheta_dtau = p_theta / Sigma;
        double dphi_dtau = (2.0 * a * r * E + (Sigma - 2.0 * r) * L / (sin(theta) * sin(theta))) / (Sigma * Delta);
        double dtau_dtau = E + (2.0 * r * (r * r + a * a) * E - 2.0 * a * r * L) / (Sigma * Delta);
        double dp_r_dtau = (1.0 / (Sigma * Delta)) * (((r * r + a * a) * mu - k) * (r - 1.0) + r * Delta * mu + 2.0 * r * (r * r + a * a) * E * E - 2.0 * a * E * L)
            - (2.0 * p_r * p_r * (r - 1.0)) / Sigma;
        double dp_theta_dtau = (sin(theta) * cos(theta)) / Sigma * ((L * L) / (sin(theta) * sin(theta)) - a * a * (E * E + mu));

        EtatGeodesique deriv;
        deriv.r = dr_dtau;
        deriv.theta = dtheta_dtau;
        deriv.phi = dphi_dtau;
        deriv.tau = dtau_dtau;
        deriv.p_r = dp_r_dtau;
        deriv.p_theta = dp_theta_dtau;

        return deriv;
        };

    // Calcul des k1 à k4
    EtatGeodesique k1 = derivatives(etat);

    EtatGeodesique etat_k2;
    etat_k2.r = etat.r + 0.5 * dt * k1.r;
    etat_k2.theta = etat.theta + 0.5 * dt * k1.theta;
    etat_k2.phi = etat.phi + 0.5 * dt * k1.phi;
    etat_k2.tau = etat.tau + 0.5 * dt * k1.tau;
    etat_k2.p_r = etat.p_r + 0.5 * dt * k1.p_r;
    etat_k2.p_theta = etat.p_theta + 0.5 * dt * k1.p_theta;
    EtatGeodesique k2 = derivatives(etat_k2);

    EtatGeodesique etat_k3;
    etat_k3.r = etat.r + 0.5 * dt * k2.r;
    etat_k3.theta = etat.theta + 0.5 * dt * k2.theta;
    etat_k3.phi = etat.phi + 0.5 * dt * k2.phi;
    etat_k3.tau = etat.tau + 0.5 * dt * k2.tau;
    etat_k3.p_r = etat.p_r + 0.5 * dt * k2.p_r;
    etat_k3.p_theta = etat.p_theta + 0.5 * dt * k2.p_theta;
    EtatGeodesique k3 = derivatives(etat_k3);

    EtatGeodesique etat_k4;
    etat_k4.r = etat.r + dt * k3.r;
    etat_k4.theta = etat.theta + dt * k3.theta;
    etat_k4.phi = etat.phi + dt * k3.phi;
    etat_k4.tau = etat.tau + dt * k3.tau;
    etat_k4.p_r = etat.p_r + dt * k3.p_r;
    etat_k4.p_theta = etat.p_theta + dt * k3.p_theta;
    EtatGeodesique k4 = derivatives(etat_k4);

    // Combiner pour obtenir le nouvel état
    EtatGeodesique nouvelEtat;
    nouvelEtat.r = etat.r + (dt / 6.0) * (k1.r + 2.0 * k2.r + 2.0 * k3.r + k4.r);
    nouvelEtat.theta = etat.theta + (dt / 6.0) * (k1.theta + 2.0 * k2.theta + 2.0 * k3.theta + k4.theta);
    nouvelEtat.phi = etat.phi + (dt / 6.0) * (k1.phi + 2.0 * k2.phi + 2.0 * k3.phi + k4.phi);
    nouvelEtat.tau = etat.tau + (dt / 6.0) * (k1.tau + 2.0 * k2.tau + 2.0 * k3.tau + k4.tau);
    nouvelEtat.p_r = etat.p_r + (dt / 6.0) * (k1.p_r + 2.0 * k2.p_r + 2.0 * k3.p_r + k4.p_r);
    nouvelEtat.p_theta = etat.p_theta + (dt / 6.0) * (k1.p_theta + 2.0 * k2.p_theta + 2.0 * k3.p_theta + k4.p_theta);

    return nouvelEtat;
}

// Variables de caméra
float distance_camera = 100.0f;
float yaw_camera = 0.0f;
float pitch_camera = 0.0f;

// Variables pour la gestion des entrées de la souris
bool souris_gauche_pressee = false;
double derniere_x = 0.0, derniere_y = 0.0;
float sensibilite = 0.005f;

// Callback pour les boutons de souris
void callback_bouton_souris(GLFWwindow* window, int bouton, int action, int mods) {
    if (bouton == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            souris_gauche_pressee = true;
            glfwGetCursorPos(window, &derniere_x, &derniere_y);
        }
        else if (action == GLFW_RELEASE) {
            souris_gauche_pressee = false;
        }
    }
}

// Callback pour le mouvement de la souris
void callback_position_souris(GLFWwindow* window, double xpos, double ypos) {
    if (souris_gauche_pressee) {
        double delta_x = xpos - derniere_x;
        double delta_y = ypos - derniere_y;
        derniere_x = xpos;
        derniere_y = ypos;

        // Mettre à jour les angles de la caméra
        yaw_camera += static_cast<float>(delta_x) * sensibilite;
        pitch_camera += static_cast<float>(delta_y) * sensibilite;

        // Limiter l'angle de pitch
        if (pitch_camera > PI / 2.0f - 0.1f)
            pitch_camera = PI / 2.0f - 0.1f;
        if (pitch_camera < -PI / 2.0f + 0.1f)
            pitch_camera = -PI / 2.0f + 0.1f;
    }
}

// Callback pour la molette de la souris
void callback_molette(GLFWwindow* window, double xoffset, double yoffset) {
    distance_camera -= static_cast<float>(yoffset) * 5.0f;
    // Limiter la distance de la caméra sans utiliser clamp
    if (distance_camera < 10.0f)
        distance_camera = 10.0f;
    if (distance_camera > 500.0f)
        distance_camera = 500.0f;
}

// Fonction pour dessiner l'ergosphère
void dessinerErgosphere(double M, double a, int slices = 50, int stacks = 50) {
    GLfloat ambient[] = { 0.0f, 0.5f, 0.0f, 0.3f };
    GLfloat diffuse[] = { 0.0f, 1.0f, 0.0f, 0.3f };
    GLfloat specular[] = { 0.0f, 0.0f, 0.0f, 1.0f };
    GLfloat shininess[] = { 0.0f };

    glMaterialfv(GL_FRONT, GL_AMBIENT, ambient);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse);
    glMaterialfv(GL_FRONT, GL_SPECULAR, specular);
    glMaterialfv(GL_FRONT, GL_SHININESS, shininess);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    for (int i = 0; i <= stacks; ++i) {
        double theta0 = PI * (static_cast<double>(i - 1) / stacks);
        double theta1 = PI * (static_cast<double>(i) / stacks);

        double r_ergo0 = M + sqrt(M * M - a * a * cos(theta0) * cos(theta0));
        double r_ergo1 = M + sqrt(M * M - a * a * cos(theta1) * cos(theta1));

        glBegin(GL_QUAD_STRIP);
        for (int j = 0; j <= slices; ++j) {
            double phi = 2.0 * PI * (static_cast<double>(j) / slices);
            double x0 = r_ergo0 * sin(theta0) * cos(phi);
            double y0 = r_ergo0 * sin(theta0) * sin(phi);
            double z0 = r_ergo0 * cos(theta0);

            double x1 = r_ergo1 * sin(theta1) * cos(phi);
            double y1 = r_ergo1 * sin(theta1) * sin(phi);
            double z1 = r_ergo1 * cos(theta1);

            // Calcul des normales
            glm::vec3 normal0 = glm::normalize(glm::vec3(x0, y0, z0));
            glm::vec3 normal1 = glm::normalize(glm::vec3(x1, y1, z1));

            glNormal3f(normal0.x, normal0.y, normal0.z);
            glVertex3d(x0, y0, z0);
            glNormal3f(normal1.x, normal1.y, normal1.z);
            glVertex3d(x1, y1, z1);
        }
        glEnd();
    }

    glDisable(GL_BLEND);
}

// Fonction pour dessiner le disque d'accrétion
void dessinerDisqueAccretion(double M, double a, double rayon_interne = 3.0, double rayon_externe = 15.0, int segments = 200) {
    GLfloat ambient_disk[] = { 1.0f, 0.5f, 0.0f, 0.7f };
    GLfloat diffuse_disk[] = { 1.0f, 0.5f, 0.0f, 0.7f };
    GLfloat specular_disk[] = { 0.0f, 0.0f, 0.0f, 1.0f };
    GLfloat shininess_disk[] = { 0.0f };

    glMaterialfv(GL_FRONT, GL_AMBIENT, ambient_disk);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse_disk);
    glMaterialfv(GL_FRONT, GL_SPECULAR, specular_disk);
    glMaterialfv(GL_FRONT, GL_SHININESS, shininess_disk);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    for (int i = 0; i < segments; ++i) {
        double theta = 2.0 * PI * (static_cast<double>(i) / segments);
        double theta_next = 2.0 * PI * (static_cast<double>(i + 1) / segments);

        double x_interne = rayon_interne * cos(theta);
        double y_interne = rayon_interne * sin(theta);
        double x_externe = rayon_externe * cos(theta);
        double y_externe = rayon_externe * sin(theta);

        double x_interne_next = rayon_interne * cos(theta_next);
        double y_interne_next = rayon_interne * sin(theta_next);
        double x_externe_next = rayon_externe * cos(theta_next);
        double y_externe_next = rayon_externe * sin(theta_next);

        // Ajout d'une légère variation en z
        double z_interne = 0.0;
        double z_externe = 0.0;
        double z_interne_next = 0.05;
        double z_externe_next = -0.05;

        // Calcul des normales
        glm::vec3 normal_interne(cos(theta), sin(theta), 0.0f);
        glm::vec3 normal_externe(cos(theta), sin(theta), 0.0f);
        glm::vec3 normal_interne_next(cos(theta_next), sin(theta_next), 0.0f);
        glm::vec3 normal_externe_next(cos(theta_next), sin(theta_next), 0.0f);

        // Dessiner deux triangles par segment
        glBegin(GL_TRIANGLES);
        // Triangle 1
        glNormal3f(normal_interne.x, normal_interne.y, normal_interne.z);
        glVertex3d(x_interne, y_interne, z_interne);
        glNormal3f(normal_externe.x, normal_externe.y, normal_externe.z);
        glVertex3d(x_externe, y_externe, z_externe);
        glNormal3f(normal_interne_next.x, normal_interne_next.y, normal_interne_next.z);
        glVertex3d(x_interne_next, y_interne_next, z_interne_next);

        // Triangle 2
        glNormal3f(normal_externe.x, normal_externe.y, normal_externe.z);
        glVertex3d(x_externe, y_externe, z_externe);
        glNormal3f(normal_externe_next.x, normal_externe_next.y, normal_externe_next.z);
        glVertex3d(x_externe_next, y_externe_next, z_externe_next);
        glNormal3f(normal_interne_next.x, normal_interne_next.y, normal_interne_next.z);
        glVertex3d(x_interne_next, y_interne_next, z_interne_next);
        glEnd();
    }

    glDisable(GL_BLEND);
}

// Fonction pour dessiner une particule lumineuse
void dessinerParticuleLumineuse(double x, double y, double z) {
    const int slices = 20;
    const int stacks = 20;
    const double rayon = 0.2;

    GLfloat ambient_particle[] = { 1.0f, 1.0f, 0.0f, 1.0f };
    GLfloat diffuse_particle[] = { 1.0f, 1.0f, 0.0f, 1.0f };
    GLfloat specular_particle[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    GLfloat shininess_particle[] = { 50.0f };

    glMaterialfv(GL_FRONT, GL_AMBIENT, ambient_particle);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse_particle);
    glMaterialfv(GL_FRONT, GL_SPECULAR, specular_particle);
    glMaterialfv(GL_FRONT, GL_SHININESS, shininess_particle);

    glPushMatrix();
    glTranslated(x, y, z);

    for (int i = 0; i < stacks; ++i) {
        double theta0 = PI * (static_cast<double>(i) / stacks - 0.5);
        double theta1 = PI * (static_cast<double>(i + 1) / stacks - 0.5);

        double z0 = sin(theta0) * rayon;
        double zr0 = cos(theta0) * rayon;

        double z1 = sin(theta1) * rayon;
        double zr1 = cos(theta1) * rayon;

        glBegin(GL_TRIANGLE_STRIP);
        for (int j = 0; j <= slices; ++j) {
            double phi = 2.0 * PI * (static_cast<double>(j) / slices);
            double x0 = zr0 * cos(phi);
            double y0 = zr0 * sin(phi);
            double x1 = zr1 * cos(phi);
            double y1 = zr1 * sin(phi);

            // Calcul des normales
            glm::vec3 normal0 = glm::normalize(glm::vec3(x0, y0, z0));
            glm::vec3 normal1 = glm::normalize(glm::vec3(x1, y1, z1));

            glNormal3f(normal0.x, normal0.y, normal0.z);
            glVertex3d(x0, y0, z0);
            glNormal3f(normal1.x, normal1.y, normal1.z);
            glVertex3d(x1, y1, z1);
        }
        glEnd();
    }

    glPopMatrix();
}

int main(int, char**) {
    // Initialisation de GLFW
    if (!glfwInit()) {
        cerr << "Erreur: Impossible d'initialiser GLFW\n";
        return 1;
    }

    // Configuration du profil OpenGL
    const char* glsl_version = "#version 110";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE);

    // Création de la fenêtre GLFW
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Simulateur de géodésiques de trou noir de Kerr", NULL, NULL);
    if (window == NULL) {
        cerr << "Erreur: Impossible de créer la fenêtre GLFW\n";
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // V-Sync

    // Enregistrement des callbacks
    glfwSetMouseButtonCallback(window, callback_bouton_souris);
    glfwSetCursorPosCallback(window, callback_position_souris);
    glfwSetScrollCallback(window, callback_molette);

    // Initialisation de GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        cerr << "Erreur: Impossible d'initialiser GLAD\n";
        return -1;
    }

    // Initialisation d'ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();

    // Initialisation des backends ImGui
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Activation des tests de profondeur et de l'éclairage
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    // Définition des propriétés de la lumière
    GLfloat light_position[] = { 100.0f, 100.0f, 100.0f, 1.0f };
    GLfloat light_diffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    GLfloat light_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    GLfloat light_ambient[] = { 0.2f, 0.2f, 0.2f, 1.0f };

    glLightfv(GL_LIGHT0, GL_POSITION, light_position);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);

    // Propriétés matérielles globales
    GLfloat material_shininess[] = { 50.0f };
    glMaterialfv(GL_FRONT, GL_SHININESS, material_shininess);
    glDisable(GL_COLOR_MATERIAL);
    glEnable(GL_NORMALIZE);
    glShadeModel(GL_SMOOTH);

    // Paramètres de la simulation
    double M = 1.0;          // Masse
    float a = 0.9f;          // Paramètre de spin (a < M)
    double mu = -1.0;        // Constante de normalisation
    double E = 0.935179;     // Énergie
    double L = 2.37176;      // Moment angulaire
    double Q = 3.82514;      // Constante de Carter

    // Conditions initiales
    ConditionsInitiales conditions_init;
    conditions_init.r = 7.0;
    conditions_init.theta = PI / 2.0;
    conditions_init.phi = 0.0;
    conditions_init.tau = 0.0;
    conditions_init.p_r = 0.0;
    conditions_init.p_theta = 1.9558;

    // État actuel
    EtatGeodesique etat = conditions_init;

    // Paramètres d'intégration
    double dt = 1.0;
    int steps = 100000;
    bool simulation_en_cours = false;

    // Données de trajectoire
    vector<tuple<double, double, double, double>> trajectoire;
    auto cart_init = cartesian(etat.r, etat.theta, etat.phi, a);
    trajectoire.emplace_back(get<0>(cart_init), get<1>(cart_init), get<2>(cart_init), etat.tau);

    // Paramètre de vitesse de simulation
    float vitesse_simulation = 1.0f;
    const int MAX_ETAPES_PAR_FRAME = 100;
    float accumulateur_etapes = 0.0f;

    // Variables pour détecter les changements de 'M' et 'a'
    double previous_M = M;
    double previous_a = a;

    // Boucle principale
    while (!glfwWindowShouldClose(window)) {
        // Gestion des événements
        glfwPollEvents();

        // Démarrer un nouveau frame ImGui
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Fenêtre de contrôle
        {
            ImGui::Begin("Contrôles de Simulation");

            // Constantes de mouvement
            ImGui::Text("Constantes de Mouvement");
            ImGui::InputDouble("Énergie (E)", &E, 0.01, 1.0, "%.5f");
            ImGui::InputDouble("Moment Angulaire (L)", &L, 0.01, 1.0, "%.5f");
            ImGui::InputDouble("Constante de Carter (Q)", &Q, 0.01, 1.0, "%.5f");
            ImGui::InputDouble("Constante de Normalisation (mu)", &mu, 0.01, 1.0, "%.5f");
            ImGui::Separator();

            // Conditions initiales
            ImGui::Text("Conditions Initiales");
            ImGui::InputDouble("Rayon Initial (r0)", &conditions_init.r, 0.1, 1.0, "%.5f");
            ImGui::InputDouble("Theta Initial (theta0)", &conditions_init.theta, 0.01, 0.1, "%.5f");
            ImGui::InputDouble("Phi Initial (phi0)", &conditions_init.phi, 0.01, 0.1, "%.5f");
            ImGui::InputDouble("Tau Initial (tau0)", &conditions_init.tau, 0.01, 0.1, "%.5f");
            ImGui::InputDouble("p_r Initial", &conditions_init.p_r, 0.01, 0.1, "%.5f");
            ImGui::InputDouble("p_theta Initial", &conditions_init.p_theta, 0.01, 0.1, "%.5f");
            ImGui::Separator();

            // Paramètres de simulation
            ImGui::Text("Paramètres de Simulation");
            ImGui::InputDouble("Masse (M)", &M, 0.1, 1.0, "%.3f");
            ImGui::InputDouble("Pas d'intégration (dt)", &dt, 0.1, 1.0, "%.5f");
            ImGui::InputInt("Nombre total d'étapes", &steps, 100, 1000, 100000);
            ImGui::Separator();

            // Vitesse de simulation
            ImGui::SliderFloat("Vitesse de Simulation", &vitesse_simulation, 0.1f, 10.0f, "x %.1f");
            ImGui::Separator();

            // Paramètre de Kerr 'a'
            ImGui::Text("Paramètre de Kerr");
            ImGui::SliderFloat("Spin (a)", &a, 0.0f, static_cast<float>(M), "%.3f");
            if (a > M) {
                a = static_cast<float>(M);
            }
            ImGui::Separator();

            // Contrôles
            if (ImGui::Button(simulation_en_cours ? "Pause" : "Start")) {
                simulation_en_cours = !simulation_en_cours;
            }
            ImGui::SameLine();
            if (ImGui::Button("Réinitialiser Simulation")) {
                etat = EtatGeodesique(conditions_init);
                trajectoire.clear();
                auto cart_init = cartesian(etat.r, etat.theta, etat.phi, a);
                trajectoire.emplace_back(get<0>(cart_init), get<1>(cart_init), get<2>(cart_init), etat.tau);
                steps = 100000;
                simulation_en_cours = false;
                previous_M = M;
                previous_a = a;
            }
            ImGui::SameLine();
            if (ImGui::Button("Exporter")) {
                // Exporter les données dans un fichier CSV
                ofstream fichier("geodesics_export.csv");
                if (fichier.is_open()) {
                    fichier << "x,y,z,tau\n";
                    for (const auto& point : trajectoire) {
                        fichier << fixed << setprecision(6)
                            << get<0>(point) << ","
                            << get<1>(point) << ","
                            << get<2>(point) << ","
                            << get<3>(point) << "\n";
                    }
                    fichier.close();
                    cout << "Données exportées dans 'geodesics_export.csv'\n";
                }
            }

            ImGui::Separator();

            // Contrôles de la caméra
            ImGui::Text("Contrôles de Caméra");
            ImGui::SliderFloat("Yaw (Horizontal)", &yaw_camera, -PI, PI, "%.2f rad");
            ImGui::SliderFloat("Pitch (Vertical)", &pitch_camera, -PI / 2, PI / 2, "%.2f rad");
            ImGui::SliderFloat("Distance", &distance_camera, 10.0f, 500.0f, "%.1f");
            if (ImGui::Button("Réinitialiser Caméra")) {
                distance_camera = 100.0f;
                yaw_camera = 0.0f;
                pitch_camera = 0.0f;
            }

            ImGui::Separator();

            // Afficher la position actuelle
            if (!trajectoire.empty()) {
                auto dernier_point = trajectoire.back();
                ImGui::Text("Position actuelle: (%.2f, %.2f, %.2f)", get<0>(dernier_point), get<1>(dernier_point), get<2>(dernier_point));
            }

            ImGui::End();
        }

        // Fenêtre "État Actuel"
        {
            ImGui::Begin("État Actuel");

            if (!trajectoire.empty()) {
                ImGui::Text("r       : %.5f", etat.r);
                ImGui::Text("theta  : %.5f rad", etat.theta);
                ImGui::Text("phi     : %.5f rad", etat.phi);
                ImGui::Text("tau     : %.5f", etat.tau);
                ImGui::Text("p_r     : %.5f", etat.p_r);
                ImGui::Text("p_theta : %.5f", etat.p_theta);
            }
            else {
                ImGui::Text("Simulation non démarrée.");
            }

            ImGui::End();
        }

        // Vérifier si 'M' a été modifié
        if (M != previous_M) {
            etat = EtatGeodesique(conditions_init);
            trajectoire.clear();
            auto cart_init = cartesian(etat.r, etat.theta, etat.phi, a);
            trajectoire.emplace_back(get<0>(cart_init), get<1>(cart_init), get<2>(cart_init), etat.tau);
            steps = 100000;
            simulation_en_cours = false;
            previous_M = M;

            // Ajuster la distance de la caméra
            distance_camera = 100.0f * (M / previous_M);
        }

        // Vérifier si 'a' a été modifié
        if (a != previous_a) {
            if (a > M) {
                a = static_cast<float>(M);
            }
            previous_a = a;

            etat = EtatGeodesique(conditions_init);
            trajectoire.clear();
            auto cart_init = cartesian(etat.r, etat.theta, etat.phi, a);
            trajectoire.emplace_back(get<0>(cart_init), get<1>(cart_init), get<2>(cart_init), etat.tau);
            steps = 100000;
            simulation_en_cours = false;
        }

        // Logique de simulation
        if (simulation_en_cours && steps > 0) {
            accumulateur_etapes += vitesse_simulation;
            int etapes_a_traiter = static_cast<int>(accumulateur_etapes);
            accumulateur_etapes -= etapes_a_traiter;

            // Limiter le nombre d'étapes pour éviter les surcharges
            if (etapes_a_traiter > MAX_ETAPES_PAR_FRAME) {
                etapes_a_traiter = MAX_ETAPES_PAR_FRAME;
                accumulateur_etapes = 0.0f;
            }

            for (int i = 0; i < etapes_a_traiter; ++i) {
                if (steps <= 0) {
                    simulation_en_cours = false;
                    break;
                }

                // Intégration RK4
                etat = rungeKutta4(etat, dt, a, mu, E, L, Q, M);

                // Conversion en coordonnées cartésiennes
                auto nouveau_cart = cartesian(etat.r, etat.theta, etat.phi, a);
                trajectoire.emplace_back(get<0>(nouveau_cart), get<1>(nouveau_cart), get<2>(nouveau_cart), etat.tau);

                // Décrémenter le nombre d'étapes
                steps--;

                // Vérifier les conditions d'arrêt
                double rayon_Schwarzschild = 2.0 * M;
                if (etat.r < rayon_Schwarzschild) {
                    cout << "La particule est tombée dans le trou noir.\n";
                    simulation_en_cours = false;
                    break;
                }

                if (etat.r > 1e7 * rayon_Schwarzschild) {
                    cout << "La particule s'est échappée.\n";
                    simulation_en_cours = false;
                    break;
                }
            }
        }

        // Dessiner la scène OpenGL
        ImGui::Render();
        int largeur, hauteur;
        glfwGetFramebufferSize(window, &largeur, &hauteur);
        glViewport(0, 0, largeur, hauteur);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Configuration des matrices de projection
        glm::mat4 projection = glm::perspective(
            glm::radians(45.0f),
            static_cast<float>(largeur) / static_cast<float>(hauteur),
            0.1f,
            1000.0f
        );
        glMatrixMode(GL_PROJECTION);
        glLoadMatrixf(glm::value_ptr(projection));

        // Calcul de la position de la caméra
        glm::vec3 position_camera(
            distance_camera * sin(yaw_camera) * cos(pitch_camera),
            distance_camera * sin(pitch_camera),
            distance_camera * cos(yaw_camera) * cos(pitch_camera)
        );

        // Configuration de la matrice de vue
        glm::mat4 vue = glm::lookAt(
            position_camera,
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 1.0f, 0.0f)
        );
        glMatrixMode(GL_MODELVIEW);
        glLoadMatrixf(glm::value_ptr(vue));

        // Dessiner les axes XYZ
        glDisable(GL_LIGHTING);
        glBegin(GL_LINES);
        // Axe X en Rouge
        glColor3f(1.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(10.0f, 0.0f, 0.0f);

        // Axe Y en Vert
        glColor3f(0.0f, 1.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, 10.0f, 0.0f);

        // Axe Z en Bleu
        glColor3f(0.0f, 0.0f, 1.0f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, 10.0f);
        glEnd();
        glEnable(GL_LIGHTING);

        // Propriétés matérielles pour l'horizon des événements
        GLfloat ambient_horizon[] = { 0.0f, 0.0f, 0.0f, 1.0f };
        GLfloat diffuse_horizon[] = { 0.0f, 0.0f, 0.0f, 1.0f };
        GLfloat specular_horizon[] = { 0.0f, 0.0f, 0.0f, 1.0f };
        GLfloat shininess_horizon[] = { 0.0f };

        glMaterialfv(GL_FRONT, GL_AMBIENT, ambient_horizon);
        glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse_horizon);
        glMaterialfv(GL_FRONT, GL_SPECULAR, specular_horizon);
        glMaterialfv(GL_FRONT, GL_SHININESS, shininess_horizon);

        // Dessiner l'horizon des événements
        int slices_horizon = 50;
        int stacks_horizon = 50;
        double rayon_horizon = M + sqrt(M * M - a * a);

        for (int i = 0; i <= stacks_horizon; ++i) {
            double theta0 = PI * (static_cast<double>(i - 1) / stacks_horizon);
            double theta1 = PI * (static_cast<double>(i) / stacks_horizon);

            double x0 = rayon_horizon * sin(theta0);
            double y0 = rayon_horizon * cos(theta0);
            double x1 = rayon_horizon * sin(theta1);
            double y1 = rayon_horizon * cos(theta1);

            glBegin(GL_QUAD_STRIP);
            for (int j = 0; j <= slices_horizon; ++j) {
                double phi = 2.0 * PI * (static_cast<double>(j) / slices_horizon);
                double cos_phi = cos(phi);
                double sin_phi = sin(phi);

                double xa0 = x0 * cos_phi;
                double ya0 = x0 * sin_phi;
                double za0 = y0;

                double xa1 = x1 * cos_phi;
                double ya1 = x1 * sin_phi;
                double za1 = y1;

                // Calcul des normales
                glm::vec3 normal0 = glm::normalize(glm::vec3(xa0, ya0, za0));
                glm::vec3 normal1 = glm::normalize(glm::vec3(xa1, ya1, za1));

                glNormal3f(normal0.x, normal0.y, normal0.z);
                glVertex3d(xa0, ya0, za0);
                glNormal3f(normal1.x, normal1.y, normal1.z);
                glVertex3d(xa1, ya1, za1);
            }
            glEnd();
        }

        // Dessiner l'ergosphère et le disque d'accrétion
        dessinerErgosphere(M, a, slices_horizon, stacks_horizon);
        dessinerDisqueAccretion(M, a, 3.0, 20.0, 200);

        // Dessiner la trajectoire en bleu
        glColor3f(0.0f, 0.0f, 1.0f);
        glBegin(GL_LINE_STRIP);
        for (const auto& point : trajectoire) {
            glVertex3d(get<0>(point), get<1>(point), get<2>(point));
        }
        glEnd();

        // Dessiner les points de trajectoire en rouge
        glPointSize(2.0f);
        glColor3f(1.0f, 0.0f, 0.0f);
        glBegin(GL_POINTS);
        for (const auto& point : trajectoire) {
            glVertex3d(get<0>(point), get<1>(point), get<2>(point));
        }
        glEnd();

        // Dessiner la particule lumineuse
        if (!trajectoire.empty()) {
            auto dernier_point = trajectoire.back();
            dessinerParticuleLumineuse(get<0>(dernier_point), get<1>(dernier_point), get<2>(dernier_point));
        }

        // Dessiner ImGui
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Échanger les buffers
        glfwSwapBuffers(window);
    }

    // Nettoyage
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
