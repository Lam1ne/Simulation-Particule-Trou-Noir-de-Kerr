// main.cpp
#include "imgui/imgui.h"
#include "imgui/imgui_impl_opengl3.h"
#include "imgui/imgui_impl_glfw.h"
#include <glad/glad.h> // Inclure GLAD avant GLFW
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
#include<stb_image.h>


// Définir les constantes GLM avant les includes
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

// Constants (unités géométrisées)
const double PI = 3.14159265358979323846;

// Structure to hold the initial conditions
struct InitialConditions {
    double r;
    double theta;
    double phi;
    double tau;
    double p_r;
    double p_theta;
};

// Structure to hold the state of the geodesic
struct GeodesicState {
    double r;
    double theta;
    double phi;
    double tau;
    double p_r;
    double p_theta;

    // Constructeur de conversion depuis InitialConditions
    GeodesicState(const InitialConditions& init) {
        r = init.r;
        theta = init.theta;
        phi = init.phi;
        tau = init.tau;
        p_r = init.p_r;
        p_theta = init.p_theta;
    }

    // Constructeur par défaut
    GeodesicState() : r(0), theta(0), phi(0), tau(0), p_r(0), p_theta(0) {}
};

// Function to convert spherical to Cartesian coordinates
std::tuple<double, double, double> cartesian(double r, double theta, double phi, double a) {
    double x = std::sqrt(r * r + a * a) * std::sin(theta) * std::cos(phi);
    double y = std::sqrt(r * r + a * a) * std::sin(theta) * std::sin(phi);
    double z = r * std::cos(theta);
    return std::make_tuple(x, y, z);
}

// Function to compute k1 to k4 using RK4
GeodesicState rungeKutta4(const GeodesicState& state, double dt, double a, double mu, double E, double L, double Q, double M) {
    // Helper lambda to compute derivatives
    auto derivatives = [&](const GeodesicState& s) -> GeodesicState {
        double r = s.r;
        double theta = s.theta;
        double p_r = s.p_r;
        double p_theta = s.p_theta;

        double Sigma = r * r + a * a * std::cos(theta) * std::cos(theta);
        double Delta = r * r - 2.0 * M * r + a * a;
        double k = Q + L * L + a * a * (E * E + mu);

        // Avoid division by zero
        if (Sigma < 1e-6) Sigma = 1e-6;
        if (Delta < 1e-6) Delta = 1e-6;

        // Compute derivatives
        double dr_dtau = Delta / Sigma * p_r;
        double dtheta_dtau = p_theta / Sigma;
        double dphi_dtau = (2.0 * a * r * E + (Sigma - 2.0 * r) * L / (std::sin(theta) * std::sin(theta))) / (Sigma * Delta);
        double dtau_dtau = E + (2.0 * r * (r * r + a * a) * E - 2.0 * a * r * L) / (Sigma * Delta);
        double dp_r_dtau = (1.0 / (Sigma * Delta)) * (((r * r + a * a) * mu - k) * (r - 1.0) + r * Delta * mu + 2.0 * r * (r * r + a * a) * E * E - 2.0 * a * E * L)
            - (2.0 * p_r * p_r * (r - 1.0)) / Sigma;
        double dp_theta_dtau = (std::sin(theta) * std::cos(theta)) / Sigma * ((L * L) / (std::sin(theta) * std::sin(theta)) - a * a * (E * E + mu));

        GeodesicState deriv;
        deriv.r = dr_dtau;
        deriv.theta = dtheta_dtau;
        deriv.phi = dphi_dtau;
        deriv.tau = dtau_dtau;
        deriv.p_r = dp_r_dtau;
        deriv.p_theta = dp_theta_dtau;

        return deriv;
        };

    // Compute k1
    GeodesicState k1 = derivatives(state);

    // Compute k2
    GeodesicState state_k2;
    state_k2.r = state.r + 0.5 * dt * k1.r;
    state_k2.theta = state.theta + 0.5 * dt * k1.theta;
    state_k2.phi = state.phi + 0.5 * dt * k1.phi;
    state_k2.tau = state.tau + 0.5 * dt * k1.tau;
    state_k2.p_r = state.p_r + 0.5 * dt * k1.p_r;
    state_k2.p_theta = state.p_theta + 0.5 * dt * k1.p_theta;
    GeodesicState k2 = derivatives(state_k2);

    // Compute k3
    GeodesicState state_k3;
    state_k3.r = state.r + 0.5 * dt * k2.r;
    state_k3.theta = state.theta + 0.5 * dt * k2.theta;
    state_k3.phi = state.phi + 0.5 * dt * k2.phi;
    state_k3.tau = state.tau + 0.5 * dt * k2.tau;
    state_k3.p_r = state.p_r + 0.5 * dt * k2.p_r;
    state_k3.p_theta = state.p_theta + 0.5 * dt * k2.p_theta;
    GeodesicState k3 = derivatives(state_k3);

    // Compute k4
    GeodesicState state_k4;
    state_k4.r = state.r + dt * k3.r;
    state_k4.theta = state.theta + dt * k3.theta;
    state_k4.phi = state.phi + dt * k3.phi;
    state_k4.tau = state.tau + dt * k3.tau;
    state_k4.p_r = state.p_r + dt * k3.p_r;
    state_k4.p_theta = state.p_theta + dt * k3.p_theta;
    GeodesicState k4 = derivatives(state_k4);

    // Combine to get the new state
    GeodesicState newState;
    newState.r = state.r + (dt / 6.0) * (k1.r + 2.0 * k2.r + 2.0 * k3.r + k4.r);
    newState.theta = state.theta + (dt / 6.0) * (k1.theta + 2.0 * k2.theta + 2.0 * k3.theta + k4.theta);
    newState.phi = state.phi + (dt / 6.0) * (k1.phi + 2.0 * k2.phi + 2.0 * k3.phi + k4.phi);
    newState.tau = state.tau + (dt / 6.0) * (k1.tau + 2.0 * k2.tau + 2.0 * k3.tau + k4.tau);
    newState.p_r = state.p_r + (dt / 6.0) * (k1.p_r + 2.0 * k2.p_r + 2.0 * k3.p_r + k4.p_r);
    newState.p_theta = state.p_theta + (dt / 6.0) * (k1.p_theta + 2.0 * k2.p_theta + 2.0 * k3.p_theta + k4.p_theta);

    return newState;
}

// Variables de caméra
float camera_distance = 100.0f;   // Distance de la caméra par rapport à l'origine
float camera_yaw = 0.0f;          // Angle de rotation horizontal (en radians)
float camera_pitch = 0.0f;        // Angle de rotation vertical (en radians)

// Variables pour la gestion des entrées de la souris
bool left_mouse_pressed = false;
double last_mouse_x = 0.0, last_mouse_y = 0.0;
float sensitivity = 0.005f;        // Sensibilité de la rotation de la caméra

// Callback pour les boutons de souris
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (action == GLFW_PRESS)
        {
            left_mouse_pressed = true;
            glfwGetCursorPos(window, &last_mouse_x, &last_mouse_y);
        }
        else if (action == GLFW_RELEASE)
        {
            left_mouse_pressed = false;
        }
    }
}

// Callback pour le mouvement de la souris
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (left_mouse_pressed)
    {
        double delta_x = xpos - last_mouse_x;
        double delta_y = ypos - last_mouse_y;
        last_mouse_x = xpos;
        last_mouse_y = ypos;

        // Mettre à jour les angles de la caméra
        camera_yaw += static_cast<float>(delta_x) * sensitivity;
        camera_pitch += static_cast<float>(delta_y) * sensitivity;

        // Limiter l'angle de pitch pour éviter les inversions
        if (camera_pitch > PI / 2.0f - 0.1f)
            camera_pitch = PI / 2.0f - 0.1f;
        if (camera_pitch < -PI / 2.0f + 0.1f)
            camera_pitch = -PI / 2.0f + 0.1f;
    }
}

// Callback pour la molette de la souris
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera_distance -= static_cast<float>(yoffset) * 5.0f; // Ajustez le facteur de zoom si nécessaire
    if (camera_distance < 10.0f)
        camera_distance = 10.0f;
    if (camera_distance > 500.0f)
        camera_distance = 500.0f;
}

// Function to draw the ergosphere
void drawErgosphere(double M, double a, int slices = 50, int stacks = 50) {
    // Définir les propriétés matérielles pour l'ergosphère (vert)
    GLfloat ambient[] = { 0.0f, 0.5f, 0.0f, 0.3f }; // Ambient green
    GLfloat diffuse[] = { 0.0f, 1.0f, 0.0f, 0.3f }; // Diffuse green
    GLfloat specular[] = { 0.0f, 0.0f, 0.0f, 1.0f }; // Specular none
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

            // Calculer les normales pour l'éclairage
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

// Function to draw the accretion disk with improved aesthetics
void drawAccretionDisk(double M, double a, double inner_radius = 3.0, double outer_radius = 15.0, int segments = 200) {
    // Définir les propriétés matérielles pour le disque d'accrétion (orange)
    GLfloat ambient_disk[] = { 1.0f, 0.5f, 0.0f, 0.7f }; // Ambient orange
    GLfloat diffuse_disk[] = { 1.0f, 0.5f, 0.0f, 0.7f }; // Diffuse orange
    GLfloat specular_disk[] = { 0.0f, 0.0f, 0.0f, 1.0f }; // Specular none
    GLfloat shininess_disk[] = { 0.0f };

    glMaterialfv(GL_FRONT, GL_AMBIENT, ambient_disk);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse_disk);
    glMaterialfv(GL_FRONT, GL_SPECULAR, specular_disk);
    glMaterialfv(GL_FRONT, GL_SHININESS, shininess_disk);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Dessiner le disque en utilisant GL_TRIANGLES pour plus de contrôle
    for (int i = 0; i < segments; ++i) {
        double theta = 2.0 * PI * (static_cast<double>(i) / segments);
        double theta_next = 2.0 * PI * (static_cast<double>(i + 1) / segments);

        // Calculer les positions des sommets
        double x_inner = inner_radius * cos(theta);
        double y_inner = inner_radius * sin(theta);
        double x_outer = outer_radius * cos(theta);
        double y_outer = outer_radius * sin(theta);

        double x_inner_next = inner_radius * cos(theta_next);
        double y_inner_next = inner_radius * sin(theta_next);
        double x_outer_next = outer_radius * cos(theta_next);
        double y_outer_next = outer_radius * sin(theta_next);

        // Ajouter une légère variation en z pour donner de la profondeur
        double z_inner = 0.0;
        double z_outer = 0.0;
        double z_inner_next = 0.05; // Léger relèvement
        double z_outer_next = -0.05; // Léger abaissement

        // Calculer les normales pour l'éclairage
        glm::vec3 normal_inner(cos(theta), sin(theta), 0.0f);
        glm::vec3 normal_outer(cos(theta), sin(theta), 0.0f);
        glm::vec3 normal_inner_next(cos(theta_next), sin(theta_next), 0.0f);
        glm::vec3 normal_outer_next(cos(theta_next), sin(theta_next), 0.0f);

        // Dessiner deux triangles pour chaque segment
        glBegin(GL_TRIANGLES);
        // Triangle 1
        glNormal3f(normal_inner.x, normal_inner.y, normal_inner.z);
        glVertex3d(x_inner, y_inner, z_inner);
        glNormal3f(normal_outer.x, normal_outer.y, normal_outer.z);
        glVertex3d(x_outer, y_outer, z_outer);
        glNormal3f(normal_inner_next.x, normal_inner_next.y, normal_inner_next.z);
        glVertex3d(x_inner_next, y_inner_next, z_inner_next);

        // Triangle 2
        glNormal3f(normal_outer.x, normal_outer.y, normal_outer.z);
        glVertex3d(x_outer, y_outer, z_outer);
        glNormal3f(normal_outer_next.x, normal_outer_next.y, normal_outer_next.z);
        glVertex3d(x_outer_next, y_outer_next, z_outer_next);
        glNormal3f(normal_inner_next.x, normal_inner_next.y, normal_inner_next.z);
        glVertex3d(x_inner_next, y_inner_next, z_inner_next);
        glEnd();
    }

    glDisable(GL_BLEND);
}

// Function to draw a luminous particle
void drawLuminousParticle(double x, double y, double z) {
    const int slices = 20;
    const int stacks = 20;
    const double radius = 0.2;

    // Définir les propriétés matérielles pour la particule (jaune lumineux)
    GLfloat ambient_particle[] = { 1.0f, 1.0f, 0.0f, 1.0f }; // Ambient yellow
    GLfloat diffuse_particle[] = { 1.0f, 1.0f, 0.0f, 1.0f }; // Diffuse yellow
    GLfloat specular_particle[] = { 1.0f, 1.0f, 1.0f, 1.0f }; // Specular white
    GLfloat shininess_particle[] = { 50.0f };

    glMaterialfv(GL_FRONT, GL_AMBIENT, ambient_particle);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse_particle);
    glMaterialfv(GL_FRONT, GL_SPECULAR, specular_particle);
    glMaterialfv(GL_FRONT, GL_SHININESS, shininess_particle);

    // Optionnel si les matériaux sont définis
    // glColor3f(1.0f, 1.0f, 0.0f); // Jaune

    glPushMatrix();
    glTranslated(x, y, z);

    // Dessiner une sphère en utilisant GL_TRIANGLE_STRIP
    for (int i = 0; i < stacks; ++i) {
        double theta0 = PI * (static_cast<double>(i) / stacks - 0.5);
        double theta1 = PI * (static_cast<double>(i + 1) / stacks - 0.5);

        double z0 = sin(theta0) * radius;
        double zr0 = cos(theta0) * radius;

        double z1 = sin(theta1) * radius;
        double zr1 = cos(theta1) * radius;

        glBegin(GL_TRIANGLE_STRIP);
        for (int j = 0; j <= slices; ++j) {
            double phi = 2.0 * PI * (static_cast<double>(j) / slices);
            double x0 = zr0 * cos(phi);
            double y0 = zr0 * sin(phi);
            double x1 = zr1 * cos(phi);
            double y1 = zr1 * sin(phi);

            // Calculer les normales pour l'éclairage
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

int main(int, char**)
{
    // Initial setup for GLFW
    if (!glfwInit())
    {
        std::cerr << "Erreur: Impossible d'initialiser GLFW\n";
        return 1;
    }

    // Configurer le profil OpenGL pour compatibilité avec le pipeline fixe
    const char* glsl_version = "#version 110"; // GLSL pour OpenGL 2.1
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE); // Utiliser n'importe quel profil disponible

    // Créer une fenêtre GLFW
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Simulateur de géodésiques trou noir de Kerr", NULL, NULL);
    if (window == NULL)
    {
        std::cerr << "Erreur: Impossible de créer la fenêtre GLFW\n";
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // V-Sync

    // Enregistrer les callbacks
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // Initialiser GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cerr << "Erreur: Impossible d'initialiser GLAD\n";
        return -1;
    }

    // Initialiser ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    // Configurer ImGui
    ImGui::StyleColorsDark();

    // Initialiser les backends ImGui
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Activer les tests de profondeur
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    // Activer l'éclairage
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    // Définir les propriétés de la lumière
    GLfloat light_position[] = { 100.0f, 100.0f, 100.0f, 1.0f }; // Position de la lumière
    GLfloat light_diffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f }; // Couleur diffuse
    GLfloat light_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f }; // Couleur spéculaire
    GLfloat light_ambient[] = { 0.2f, 0.2f, 0.2f, 1.0f }; // Couleur ambiante

    glLightfv(GL_LIGHT0, GL_POSITION, light_position);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);

    // Définir les propriétés matérielles globales (optionnel, car matériaux sont définis explicitement)
    GLfloat material_shininess[] = { 50.0f }; // Brillance
    glMaterialfv(GL_FRONT, GL_SHININESS, material_shininess);

    // Désactiver GL_COLOR_MATERIAL pour un contrôle total
    glDisable(GL_COLOR_MATERIAL);

    // Activer le lissage des normales et le shading lisse
    glEnable(GL_NORMALIZE);
    glShadeModel(GL_SMOOTH);

    // Simulation parameters
    double M = 1.0;          // Masse (unités géométrisées)
    float a = 0.9f;          // Paramètre de spin (a < M)
    double mu = -1.0;        // Constante de normalisation (pour géodésiques temporelles)
    double E = 0.935179;     // Énergie
    double L = 2.37176;      // Moment angulaire
    double Q = 3.82514;      // Constante de Carter

    // Initial conditions
    InitialConditions initial_conditions;
    initial_conditions.r = 7.0;
    initial_conditions.theta = PI / 2.0;
    initial_conditions.phi = 0.0;
    initial_conditions.tau = 0.0;
    initial_conditions.p_r = 0.0;
    initial_conditions.p_theta = 1.9558;

    // Current state
    GeodesicState state = initial_conditions; // Utilise le constructeur de conversion

    // Integration parameters
    double dt = 1.0;          // Pas d'intégration
    int steps = 100000;       // Nombre total d'étapes
    bool simulation_running = false;

    // Trajectory data
    std::vector<std::tuple<double, double, double, double>> trajectory;
    auto initial_cart = cartesian(state.r, state.theta, state.phi, a);
    trajectory.emplace_back(std::get<0>(initial_cart), std::get<1>(initial_cart), std::get<2>(initial_cart), state.tau);

    // Paramètre de vitesse de simulation
    float simulation_speed = 1.0f; // Facteur de vitesse (1.0 = vitesse normale)
    const int MAX_STEPS_PER_FRAME = 100; // Limite maximale d'étapes par frame

    // Variable pour accumulation des étapes fractionnelles
    float step_accumulator = 0.0f;

    // Variable pour détecter les changements de 'a'
    double previous_a = a;

    // Boucle principale
    while (!glfwWindowShouldClose(window))
    {
        // Gestion des événements
        glfwPollEvents();

        // Démarrer un nouveau frame ImGui
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Création de la fenêtre de contrôle
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
            ImGui::InputDouble("Rayon Initial (r0)", &initial_conditions.r, 0.1, 1.0, "%.5f");
            ImGui::InputDouble("Theta Initial (theta0)", &initial_conditions.theta, 0.01, 0.1, "%.5f");
            ImGui::InputDouble("Phi Initial (phi0)", &initial_conditions.phi, 0.01, 0.1, "%.5f");
            ImGui::InputDouble("Tau Initial (tau0)", &initial_conditions.tau, 0.01, 0.1, "%.5f");
            ImGui::InputDouble("p_r Initial", &initial_conditions.p_r, 0.01, 0.1, "%.5f");
            ImGui::InputDouble("p_theta Initial", &initial_conditions.p_theta, 0.01, 0.1, "%.5f");
            ImGui::Separator();

            // Paramètres de simulation
            ImGui::Text("Paramètres de Simulation");
            ImGui::InputDouble("Pas d'intégration (dt)", &dt, 0.1, 1.0, "%.5f");
            ImGui::InputInt("Nombre total d'étapes", &steps, 100, 1000, 100000);
            ImGui::Separator();

            // Contrôles de la vitesse de simulation
            ImGui::SliderFloat("Vitesse de Simulation", &simulation_speed, 0.1f, 10.0f, "x %.1f");
            ImGui::Separator();

            // Contrôles du paramètre de Kerr 'a'
            ImGui::Text("Paramètre de Kerr");
            ImGui::SliderFloat("Spin (a)", &a, 0.0f, M, "%.3f");
            if (a > M)
            {
                a = M; // Clamp 'a' to maximum value
            }
            ImGui::Separator();

            // Boutons de contrôle
            if (ImGui::Button(simulation_running ? "Pause" : "Start"))
            {
                simulation_running = !simulation_running;
            }
            ImGui::SameLine();
            if (ImGui::Button("Réinitialiser Simulation"))
            {
                // Réinitialiser les conditions initiales
                state = GeodesicState(initial_conditions); // Utilise le constructeur de conversion
                trajectory.clear();
                auto initial_cart = cartesian(state.r, state.theta, state.phi, a);
                trajectory.emplace_back(std::get<0>(initial_cart), std::get<1>(initial_cart), std::get<2>(initial_cart), state.tau);
                steps = 100000; // Réinitialiser le nombre d'étapes si nécessaire
                simulation_running = false;
                previous_a = a; // Mettre à jour la valeur précédente de 'a'
            }
            ImGui::SameLine();
            if (ImGui::Button("Exporter"))
            {
                // Exporter les données dans un fichier CSV
                std::ofstream file("geodesics_export.csv");
                if (file.is_open())
                {
                    file << "x,y,z,tau\n";
                    for (const auto& point : trajectory)
                    {
                        file << std::fixed << std::setprecision(6)
                            << std::get<0>(point) << ","
                            << std::get<1>(point) << ","
                            << std::get<2>(point) << ","
                            << std::get<3>(point) << "\n";
                    }
                    file.close();
                    std::cout << "Données exportées dans 'geodesics_export.csv'\n";
                }
                else
                {
                    std::cerr << "Erreur: Impossible d'ouvrir le fichier pour l'exportation.\n";
                }
            }

            ImGui::Separator();

            // Contrôles de la caméra
            ImGui::Text("Contrôles de Caméra");
            ImGui::SliderFloat("Yaw (Horizontal)", &camera_yaw, -PI, PI, "%.2f rad");
            ImGui::SliderFloat("Pitch (Vertical)", &camera_pitch, -PI / 2, PI / 2, "%.2f rad");
            ImGui::SliderFloat("Distance", &camera_distance, 10.0f, 500.0f, "%.1f");
            if (ImGui::Button("Réinitialiser Caméra"))
            {
                camera_distance = 100.0f;
                camera_yaw = 0.0f;
                camera_pitch = 0.0f;
            }

            ImGui::Separator();

            // Afficher la position actuelle
            if (!trajectory.empty())
            {
                auto last_point = trajectory.back();
                ImGui::Text("Position actuelle: (%.2f, %.2f, %.2f)", std::get<0>(last_point), std::get<1>(last_point), std::get<2>(last_point));
            }

            ImGui::End();
        }

        // Création de la fenêtre "État Actuel"
        {
            ImGui::Begin("État Actuel");

            if (!trajectory.empty())
            {
                ImGui::Text("r       : %.5f", state.r);
                ImGui::Text("theta   : %.5f rad", state.theta);
                ImGui::Text("phi     : %.5f rad", state.phi);
                ImGui::Text("tau     : %.5f", state.tau);
                ImGui::Text("p_r     : %.5f", state.p_r);
                ImGui::Text("p_theta : %.5f", state.p_theta);
            }
            else
            {
                ImGui::Text("Simulation non démarrée.");
            }

            ImGui::End();
        }

        // Vérifier si 'a' a été modifié en dehors des contrôles
        if (a != previous_a)
        {
            // Réinitialiser les conditions initiales pour maintenir la cohérence
            state = GeodesicState(initial_conditions); // Utilise le constructeur de conversion
            trajectory.clear();
            auto initial_cart = cartesian(state.r, state.theta, state.phi, a);
            trajectory.emplace_back(std::get<0>(initial_cart), std::get<1>(initial_cart), std::get<2>(initial_cart), state.tau);
            steps = 100000; // Réinitialiser le nombre d'étapes si nécessaire
            simulation_running = false;
            previous_a = a;
        }

        // Simulation logic
        if (simulation_running && steps > 0)
        {
            // Accumuler les étapes basées sur la vitesse
            step_accumulator += simulation_speed;

            // Calculer le nombre d'étapes à effectuer cette frame
            int steps_to_process = static_cast<int>(step_accumulator);
            step_accumulator -= steps_to_process;

            // Limiter le nombre d'étapes pour éviter les surcharges
            if (steps_to_process > MAX_STEPS_PER_FRAME)
            {
                steps_to_process = MAX_STEPS_PER_FRAME;
                step_accumulator = 0.0f;
            }

            for (int i = 0; i < steps_to_process; ++i)
            {
                if (steps <= 0)
                {
                    simulation_running = false;
                    break;
                }

                // Effectuer un pas d'intégration RK4
                state = rungeKutta4(state, dt, a, mu, E, L, Q, M);

                // Convertir en coordonnées cartésiennes
                auto new_cart = cartesian(state.r, state.theta, state.phi, a);
                trajectory.emplace_back(std::get<0>(new_cart), std::get<1>(new_cart), std::get<2>(new_cart), state.tau);

                // Décrémenter le nombre total d'étapes
                steps--;

                // Vérifier les conditions d'arrêt
                double Schwarzschild_radius = 2.0 * M; // Unités géométrisées
                if (state.r < Schwarzschild_radius)
                {
                    std::cout << "La particule est tombée dans le trou noir.\n";
                    simulation_running = false;
                    break;
                }

                if (state.r > 1e7 * Schwarzschild_radius)
                {
                    std::cout << "La particule s'est échappée.\n";
                    simulation_running = false;
                    break;
                }
            }
        }

        // Dessiner la scène OpenGL
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Configurer les matrices de projection
        glm::mat4 projection = glm::perspective(
            glm::radians(45.0f),                                // Champ de vision (FOV)
            static_cast<float>(display_w) / static_cast<float>(display_h), // Rapport d'aspect
            0.1f,                                               // Plan proche
            1000.0f                                             // Plan éloigné
        );
        glMatrixMode(GL_PROJECTION);
        glLoadMatrixf(glm::value_ptr(projection));

        // Calculer la position de la caméra basée sur yaw, pitch et distance
        glm::vec3 camera_position(
            camera_distance * sin(camera_yaw) * cos(camera_pitch),
            camera_distance * sin(camera_pitch),
            camera_distance * cos(camera_yaw) * cos(camera_pitch)
        );

        // Configurer la matrice de vue avec la nouvelle position
        glm::mat4 view = glm::lookAt(
            camera_position,                // Position de la caméra
            glm::vec3(0.0f, 0.0f, 0.0f),    // Point vers lequel la caméra regarde
            glm::vec3(0.0f, 1.0f, 0.0f)     // Vecteur "up"
        );
        glMatrixMode(GL_MODELVIEW);
        glLoadMatrixf(glm::value_ptr(view));

        // Dessiner les axes XYZ sans influence de l'éclairage
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


        GLfloat ambient_horizon[] = { 0.0f, 0.0f, 0.0f, 1.0f };    // Couleur ambiante noire
        GLfloat diffuse_horizon[] = { 0.0f, 0.0f, 0.0f, 1.0f };    // Couleur diffuse noire
        GLfloat specular_horizon[] = { 0.0f, 0.0f, 0.0f, 1.0f };   // Pas de composante spéculaire
        GLfloat shininess_horizon[] = { 0.0f };                    // Brillance nulle pour un aspect mat

        glMaterialfv(GL_FRONT, GL_AMBIENT, ambient_horizon);
        glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse_horizon);
        glMaterialfv(GL_FRONT, GL_SPECULAR, specular_horizon);
        glMaterialfv(GL_FRONT, GL_SHININESS, shininess_horizon);

        // Dessiner l'horizon des événements (trou noir) en gris clair
        // Déjà défini via matériel, donc aucune modification de couleur ici

        int slices = 50;
        int stacks = 50;
        double horizon_radius = M + sqrt(M * M - a * a); // Rayon de l'horizon des événements

        for (int i = 0; i <= stacks; ++i) {
            double theta0 = PI * (static_cast<double>(i - 1) / stacks);
            double theta1 = PI * (static_cast<double>(i) / stacks);

            double x0 = horizon_radius * sin(theta0);
            double y0 = horizon_radius * cos(theta0);
            double x1 = horizon_radius * sin(theta1);
            double y1 = horizon_radius * cos(theta1);

            glBegin(GL_QUAD_STRIP);
            for (int j = 0; j <= slices; ++j) {
                double phi = 2.0 * PI * (static_cast<double>(j) / slices);
                double cos_phi = cos(phi);
                double sin_phi = sin(phi);

                double x_a0 = x0 * cos_phi;
                double y_a0 = x0 * sin_phi;
                double z_a0 = y0;

                double x_a1 = x1 * cos_phi;
                double y_a1 = x1 * sin_phi;
                double z_a1 = y1;

                // Calculer les normales pour l'éclairage
                glm::vec3 normal0 = glm::normalize(glm::vec3(x_a0, y_a0, z_a0));
                glm::vec3 normal1 = glm::normalize(glm::vec3(x_a1, y_a1, z_a1));

                glNormal3f(normal0.x, normal0.y, normal0.z);
                glVertex3d(x_a0, y_a0, z_a0);
                glNormal3f(normal1.x, normal1.y, normal1.z);
                glVertex3d(x_a1, y_a1, z_a1);
            }
            glEnd();
        }

        // Dessiner l'ergosphère en vert
        drawErgosphere(M, a, slices, stacks);

        // Dessiner le disque d'accrétion en orange
        drawAccretionDisk(M, a, 3.0, 20.0, 200); // Ajuster les rayons et segments selon besoin

        // Dessiner la trajectoire en rouge
        // Déjà défini via matériel
        glColor3f(1.0f, 0.0f, 0.0f); // Rouge
        glBegin(GL_LINE_STRIP);
        for (const auto& point : trajectory)
        {
            glVertex3d(std::get<0>(point), std::get<1>(point), std::get<2>(point));
        }
        glEnd();

        // Dessiner les points de trajectoire (optionnel)
        glPointSize(2.0f);
        glColor3f(1.0f, 0.0f, 0.0f); // Rouge
        glBegin(GL_POINTS);
        for (const auto& point : trajectory)
        {
            glVertex3d(std::get<0>(point), std::get<1>(point), std::get<2>(point));
        }
        glEnd();

        // Dessiner la petite boule lumineuse pour la particule en jaune
        if (!trajectory.empty())
        {
            auto last_point = trajectory.back();
            drawLuminousParticle(std::get<0>(last_point), std::get<1>(last_point), std::get<2>(last_point));
        }

        // Dessiner ImGui
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Swap buffers
        glfwSwapBuffers(window);

        // Vérifier les erreurs OpenGL (optionnel)
        GLenum err;
        while ((err = glGetError()) != GL_NO_ERROR) {
            std::cerr << "Erreur OpenGL : " << err << std::endl;
        }
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
