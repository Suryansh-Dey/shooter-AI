#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include "functions.cxx"
#include "animation.cxx"
#include "shooter.cxx"
#include "AI.cxx"
#include "UI.cxx"
//**** ADJUSTABLE CONSTANT PARAMETERS ****
constexpr int FPS = 30;
//**** GAME CONSTANTS ****
constexpr int FRAME_GAP = 1000 / FPS;
std::unordered_map<std::string, SDL_Texture *> shooterImages, deathImages, buttonImages;
int SCREEN_WIDTH, SCREEN_HEIGHT;

//**** VARIABLES ****
double shootedBullet = -1;

int main(int argc, char* argv[])
{
	SDL_Init(SDL_INIT_EVERYTHING);
	SDL_Renderer *s = createWindow(SCREEN_WIDTH, SCREEN_HEIGHT);
	//**** LOADING RESOURCES ****
	shooterImages["shooter"] = IMG_LoadTexture(s, "resources/shooter.png");
	shooterImages["gun"] = IMG_LoadTexture(s, "resources/wizard.png");
	shooterImages["snowBall"] = IMG_LoadTexture(s, "resources/snowBall.png");
	shooterImages["fireBall"] = IMG_LoadTexture(s, "resources/fireBall.png");
	shooterImages["web"] = IMG_LoadTexture(s, "resources/web.png");
	shooterImages["snowParticle_1"] = IMG_LoadTexture(s, "resources/snowParticle_1.png");
	shooterImages["snowParticle_2"] = IMG_LoadTexture(s, "resources/snowParticle_2.png");
	shooterImages["fireParticle_1"] = IMG_LoadTexture(s, "resources/fireParticle_1.png");
	shooterImages["fireParticle_2"] = IMG_LoadTexture(s, "resources/fireParticle_2.png");
	shooterImages["magic_1"] = IMG_LoadTexture(s, "resources/magic_1.png");
	shooterImages["magic_2"] = IMG_LoadTexture(s, "resources/magic_2.png");
	shooterImages["magic_3"] = IMG_LoadTexture(s, "resources/magic_3.png");
	shooterImages["snowBurst"] = IMG_LoadTexture(s, "resources/snowBurst.png");
	shooterImages["fireBurst"] = IMG_LoadTexture(s, "resources/fireBurst.png");
	shooterImages["spark_1"] = IMG_LoadTexture(s, "resources/spark_1.png");
	shooterImages["spark_2"] = IMG_LoadTexture(s, "resources/spark_2.png");
	shooterImages["frost"] = IMG_LoadTexture(s, "resources/frost.png");
	deathImages["gunDeath_1"] = IMG_LoadTexture(s, "resources/wizardDeath_1.png");
	deathImages["gunDeath_2"] = IMG_LoadTexture(s, "resources/wizardDeath_2.png");
	deathImages["gunDeath_3"] = IMG_LoadTexture(s, "resources/wizardDeath_3.png");
	buttonImages["snowBall"] = IMG_LoadTexture(s, "resources/snowBall.png");
	buttonImages["fireBall"] = IMG_LoadTexture(s, "resources/fireBall.png");
	buttonImages["web"] = IMG_LoadTexture(s, "resources/web.png");
	buttonImages["joystickPad"] = IMG_LoadTexture(s, "resources/joystickPad.png");
	buttonImages["joystickButton"] = IMG_LoadTexture(s, "resources/joystickButton.png");
	buttonImages["train"] = IMG_LoadTexture(s, "resources/train.png");
	buttonImages["select"] = IMG_LoadTexture(s, "resources/select.png");

	Shooter player1(SCREEN_WIDTH, SCREEN_HEIGHT, 100, 100, 2, shooterImages);
	Shooter player2(SCREEN_WIDTH, SCREEN_HEIGHT, 100, 700, 2, shooterImages);
	Brain ai(2, SCREEN_WIDTH, SCREEN_HEIGHT);
	InputManager inputManager(player1, ai, shootedBullet, buttonImages, SCREEN_WIDTH, SCREEN_HEIGHT);
	//**** GAME LOOP ****
	bool player1Alive = true, player2Alive = true;
	while (inputManager.handelInput())
	{
		ai.invokeAI(player2, player1);
		//ai.memorize(player1, player2, shootedBullet);
		player1.update();
		player2.update();
		collisionHandeler(player1, player2);
		if (not player1.isAlive() && player1Alive)
		{
			animateDeath(player1, s, deathImages);
			player1.respawn(shooterImages);
		}
		else if (not player2.isAlive() && player2Alive)
		{
			animateDeath(player2, s, deathImages);
			player2.respawn(shooterImages);
		}
		player1.render(s);
		player2.render(s);
		inputManager.render(s);
		SDL_RenderPresent(s);
		SDL_RenderClear(s);
		FPS_manager(FRAME_GAP);
	}
	return 0;
}