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

// Forcer l'utilisation des radians et du système de profondeur de 0 à 1
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
EtatGeodesique rungeKutta4(const EtatGeodesique& state, double dt, double a, double mu, double E, double L, double Q, double M) {
    auto derivees = [&](const EtatGeodesique& s) -> EtatGeodesique {
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
        double dp_r_dtau = (1.0 / (Sigma * Delta)) * (((r * r + a * a) * mu - k) * (r - M) + r * Delta * mu + 2.0 * r * (r * r + a * a) * E * E - 2.0 * a * E * L)
            - (2.0 * p_r * p_r * (r - M)) / Sigma;
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
    EtatGeodesique k1 = derivees(state);

    EtatGeodesique state_k2;
    state_k2.r = state.r + 0.5 * dt * k1.r;
    state_k2.theta = state.theta + 0.5 * dt * k1.theta;
    state_k2.phi = state.phi + 0.5 * dt * k1.phi;
    state_k2.tau = state.tau + 0.5 * dt * k1.tau;
    state_k2.p_r = state.p_r + 0.5 * dt * k1.p_r;
    state_k2.p_theta = state.p_theta + 0.5 * dt * k1.p_theta;
    EtatGeodesique k2 = derivees(state_k2);

    EtatGeodesique state_k3;
    state_k3.r = state.r + 0.5 * dt * k2.r;
    state_k3.theta = state.theta + 0.5 * dt * k2.theta;
    state_k3.phi = state.phi + 0.5 * dt * k2.phi;
    state_k3.tau = state.tau + 0.5 * dt * k2.tau;
    state_k3.p_r = state.p_r + 0.5 * dt * k2.p_r;
    state_k3.p_theta = state.p_theta + 0.5 * dt * k2.p_theta;
    EtatGeodesique k3 = derivees(state_k3);

    EtatGeodesique state_k4;
    state_k4.r = state.r + dt * k3.r;
    state_k4.theta = state.theta + dt * k3.theta;
    state_k4.phi = state.phi + dt * k3.phi;
    state_k4.tau = state.tau + dt * k3.tau;
    state_k4.p_r = state.p_r + dt * k3.p_r;
    state_k4.p_theta = state.p_theta + dt * k3.p_theta;
    EtatGeodesique k4 = derivees(state_k4);

    // Combiner pour obtenir le nouvel état
    EtatGeodesique nouvelEtat;
    nouvelEtat.r = state.r + (dt / 6.0) * (k1.r + 2.0 * k2.r + 2.0 * k3.r + k4.r);
    nouvelEtat.theta = state.theta + (dt / 6.0) * (k1.theta + 2.0 * k2.theta + 2.0 * k3.theta + k4.theta);
    nouvelEtat.phi = state.phi + (dt / 6.0) * (k1.phi + 2.0 * k2.phi + 2.0 * k3.phi + k4.phi);
    nouvelEtat.tau = state.tau + (dt / 6.0) * (k1.tau + 2.0 * k2.tau + 2.0 * k3.tau + k4.tau);
    nouvelEtat.p_r = state.p_r + (dt / 6.0) * (k1.p_r + 2.0 * k2.p_r + 2.0 * k3.p_r + k4.p_r);
    nouvelEtat.p_theta = state.p_theta + (dt / 6.0) * (k1.p_theta + 2.0 * k2.p_theta + 2.0 * k3.p_theta + k4.p_theta);

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
    // Limiter la distance de la caméra
    if (distance_camera < 10.0f)
        distance_camera = 10.0f;
    if (distance_camera > 500.0f)
        distance_camera = 500.0f;
}

// Fonction pour dessiner l'ergosphère
void dessinerErgosphere(double M, double a, int slices = 50, int stacks = 50) {
    // Votre implémentation existante
    // ...
}

// Fonction pour dessiner le disque d'accrétion
void dessinerDisqueAccretion(double M, double a, double rayon_interne = 3.0, double rayon_externe = 15.0, int segments = 200) {
    // Votre implémentation existante
    // ...
}

// Déclaration de la variable pour contrôler l'affichage du disque d'accrétion
bool afficher_disque_accretion = true;

int main() {
    // Initialisation de GLFW
    if (!glfwInit()) {
        cerr << "Échec de l'initialisation de GLFW" << endl;
        return -1;
    }

    // Configuration de la fenêtre GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Création de la fenêtre GLFW
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Simulation de Trajectoire autour d'un Trou Noir de Kerr", nullptr, nullptr);
    if (!window) {
        cerr << "Échec de la création de la fenêtre GLFW" << endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetMouseButtonCallback(window, callback_bouton_souris);
    glfwSetCursorPosCallback(window, callback_position_souris);
    glfwSetScrollCallback(window, callback_molette);

    // Initialisation de GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        cerr << "Échec de l'initialisation de GLAD" << endl;
        return -1;
    }

    glEnable(GL_DEPTH_TEST);

    // Configure les options d'OpenGL
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    // Initialisation d'ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // Variables physiques
    double M = 1.0;   // Masse du trou noir
    double a = 0.5;   // Paramètre de rotation (0 <= a <= M)
    double mu = 0.0;  // Masse de la particule (0 pour photon)
    double E = 1.0;   // Énergie spécifique
    double L = 2.0;   // Moment angulaire spécifique
    double Q = 0.0;   // Constante de Carter

    // Conditions initiales
    ConditionsInitiales initState;
    initState.r = 10.0;
    initState.theta = PI / 2.0;
    initState.phi = 0.0;
    initState.tau = 0.0;
    initState.p_r = 0.0;
    initState.p_theta = 0.0;

    // Paramètres de simulation
    double dt = 0.01;
    int steps = 10000;

    // Vecteur pour stocker la trajectoire
    vector<tuple<double, double, double>> trajectoire;

    // Variables pour la simulation
    bool simuler = false;

    // Boucle principale
    while (!glfwWindowShouldClose(window)) {
        // Gestion des événements
        glfwPollEvents();

        // Démarre un nouveau frame ImGui
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Interface ImGui
        ImGui::Begin("Paramètres de Simulation");
        ImGui::Text("Paramètres du Trou Noir");
        ImGui::InputDouble("Masse (M)", &M, 0.1, 1.0, "%.5f");
        ImGui::InputDouble("Rotation (a)", &a, 0.01, 0.1, "%.5f");
        ImGui::Separator();

        ImGui::Text("Paramètres de la Particule");
        ImGui::InputDouble("Masse (mu)", &mu, 0.0, 0.1, "%.5f");
        ImGui::InputDouble("Énergie Spécifique (E)", &E, 0.1, 1.0, "%.5f");
        ImGui::InputDouble("Moment Angulaire Spécifique (L)", &L, 0.1, 1.0, "%.5f");
        ImGui::InputDouble("Constante de Carter (Q)", &Q, 0.1, 1.0, "%.5f");
        ImGui::Separator();

        ImGui::Text("Conditions Initiales");
        ImGui::InputDouble("Rayon Initial (r)", &initState.r, 0.1, 1.0, "%.5f");
        ImGui::InputDouble("Angle Theta Initial (theta)", &initState.theta, 0.01, 0.1, "%.5f");
        ImGui::InputDouble("Angle Phi Initial (phi)", &initState.phi, 0.01, 0.1, "%.5f");
        ImGui::Separator();

        ImGui::Text("Paramètres de Simulation");
        ImGui::InputDouble("Pas d'Intégration (dt)", &dt, 0.001, 0.01, "%.5f");
        ImGui::InputInt("Nombre d'Étapes", &steps);
        ImGui::Separator();

        // Ajout de la case à cocher pour le disque d'accrétion
        ImGui::Checkbox("Afficher le disque d'accrétion", &afficher_disque_accretion);
        ImGui::Separator();

        if (ImGui::Button("Démarrer la Simulation")) {
            simuler = true;
            trajectoire.clear();

            // Initialiser l'état géodésique
            EtatGeodesique state(initState);

            // Effectuer la simulation
            for (int i = 0; i < steps; ++i) {
                state = rungeKutta4(state, dt, a, mu, E, L, Q, M);
                auto [x, y, z] = cartesian(state.r, state.theta, state.phi, a);
                trajectoire.push_back(make_tuple(x, y, z));
            }
        }
        ImGui::End();

        // Préparation du rendu
        glClearColor(0.0f, 0.0f, 0.0f, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Configuration de la matrice de projection
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1280.0f / 720.0f, 0.1f, 1000.0f);

        // Configuration de la matrice de vue
        glm::vec3 cameraPos = glm::vec3(
            distance_camera * cos(pitch_camera) * sin(yaw_camera),
            distance_camera * sin(pitch_camera),
            distance_camera * cos(pitch_camera) * cos(yaw_camera)
        );
        glm::mat4 view = glm::lookAt(cameraPos, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));

        // Application des matrices
        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 mvp = projection * view * model;
        glMatrixMode(GL_MODELVIEW);
        glLoadMatrixf(glm::value_ptr(mvp));

        // Dessiner l'ergosphère
        dessinerErgosphere(M, a);

        // Dessiner le disque d'accrétion si activé
        if (afficher_disque_accretion) {
            dessinerDisqueAccretion(M, a);
        }

        // Dessiner la trajectoire
        if (simuler) {
            glColor3f(1.0f, 1.0f, 0.0f);
            glBegin(GL_LINE_STRIP);
            for (const auto& [x, y, z] : trajectoire) {
                glVertex3d(x, y, z);
            }
            glEnd();
        }

        // Rendu d'ImGui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Échange des buffers
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
