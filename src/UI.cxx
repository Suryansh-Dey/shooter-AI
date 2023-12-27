SDL_Renderer *createWindow(int &SCREEN_WIDTH, int &SCREEN_HEIGHT)
{
	SDL_Window *window = SDL_CreateWindow("Shooter AI",SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,0,0,SDL_WINDOW_FULLSCREEN);
	SDL_GetWindowSize(window,&SCREEN_WIDTH,&SCREEN_HEIGHT);
	SDL_Renderer *s = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
	SDL_RenderClear(s);
	return s;
}
class InputManager
{
	Shooter &player;
	Brain &ai;
	double &shootedBullet;
	const int SCREEN_WIDTH, SCREEN_HEIGHT;
	struct Finger
	{
		bool available = 0;
		float birth_x, birth_y;
		float current_x, current_y;
	};
	struct Button
	{
		SDL_Texture *image;
		SDL_Rect rect;
		float x, y, w, h;
		bool clicked = 0;
		Button() {}
		Button(SDL_Texture *arg_image, float arg_x, float arg_y, float side, int SCREEN_WIDTH, int SCREEN_HEIGHT) : image(arg_image), x(arg_x), y(arg_y)
		{ //setting button size proportional to screen width
			w = side;
			h = side * SCREEN_WIDTH / SCREEN_HEIGHT;
			rect = createRect(x * SCREEN_WIDTH, y * SCREEN_HEIGHT, w * SCREEN_WIDTH, h * SCREEN_HEIGHT);
		}
	};
	struct Joystick
	{
		bool clicked = 0;
		SDL_Texture *padImage, *buttonImage;
		SDL_Rect padRect, buttonRect;
	};
	static constexpr int MAX_FINGER_COUNT = 5;
	struct Finger inputs[MAX_FINGER_COUNT];
	std::unordered_map<std::string, Button> buttons;
	struct Joystick joystick;

	bool updateInputs(struct Finger inputs[MAX_FINGER_COUNT]);
	inline bool isClicked(Button button, struct Finger input);

  public:
	InputManager(Shooter &arg_player, Brain &ai, double &shootedBullet, std::unordered_map<std::string, SDL_Texture *> images, int arg_SCREEN_WIDTH, int arg_SCREEN_HEIGHT) : player(arg_player), ai(ai), shootedBullet(shootedBullet), SCREEN_WIDTH(arg_SCREEN_WIDTH), SCREEN_HEIGHT(arg_SCREEN_HEIGHT)
	{
		buttons["snowBall"] = Button(images["snowBall"], 0.9, 0.9, 0.1, SCREEN_WIDTH, SCREEN_HEIGHT);
		buttons["fireBall"] = Button(images["fireBall"], 0.9, 0.7, 0.1, SCREEN_WIDTH, SCREEN_HEIGHT);
		buttons["web"] = Button(images["web"], 0.9, 0.5, 0.1, SCREEN_WIDTH, SCREEN_HEIGHT);
		buttons["train"] = Button(images["train"], 0.1, 0.1, 0.1, SCREEN_WIDTH, SCREEN_HEIGHT);
		buttons["select"] = Button(images["select"], 0.1, 0.3, 0.1, SCREEN_WIDTH, SCREEN_HEIGHT);
		joystick = Joystick{0, images["joystickPad"], images["joystickButton"], createRect(200, 500, 100, 100), SDL_Rect{0, 0, 50, 50}};
	}
	bool handelInput();
	void render(SDL_Renderer *s);
};

bool InputManager::updateInputs(struct Finger inputs[MAX_FINGER_COUNT])
{
	SDL_Event event;
	while (SDL_PollEvent(&event))
	{
		if (event.type == SDL_FINGERDOWN && event.tfinger.fingerId < MAX_FINGER_COUNT)
		{
			inputs[event.tfinger.fingerId].available = true;
			inputs[event.tfinger.fingerId].birth_x = event.tfinger.x;
			inputs[event.tfinger.fingerId].birth_y = event.tfinger.y;
			inputs[event.tfinger.fingerId].current_x = event.tfinger.x;
			inputs[event.tfinger.fingerId].current_y = event.tfinger.y;
		}
		else if (event.type == SDL_FINGERUP && event.tfinger.fingerId < MAX_FINGER_COUNT)
			inputs[event.tfinger.fingerId].available = false;
		else if (event.type == SDL_FINGERMOTION && event.tfinger.fingerId < MAX_FINGER_COUNT)
		{
			inputs[event.tfinger.fingerId].current_x += event.tfinger.dx;
			inputs[event.tfinger.fingerId].current_y += event.tfinger.dy;
		}
		else if (event.type == SDL_QUIT)
			return false;
	}
	return true;
}
inline bool InputManager::isClicked(Button button, struct Finger input)
{
	return abs(button.x - input.birth_x) < button.w / 2 and abs(button.y - input.birth_y) < button.h / 2;
}
bool InputManager::handelInput()
{
	if (!updateInputs(inputs))
		return false;
	int movementAngle, shootingAngle = INT_MAX;
	joystick.clicked = 0;
	buttons["train"].clicked = false;
	buttons["select"].clicked = false;
	for (struct Finger &input : inputs)
	{
		if (!input.available)
			continue;
		if (input.birth_x < 0.5 && input.birth_y > 0.5 && input.birth_x != input.current_x)
		{
			movementAngle = angleOfVector(input.current_x - input.birth_x, input.current_y - input.birth_y);
			joystick.clicked = 1;
			joystick.padRect = createRect(input.birth_x * SCREEN_WIDTH, input.birth_y * SCREEN_HEIGHT, joystick.padRect.w, joystick.padRect.h);
		}
		if (isClicked(buttons["fireBall"], input))
		{
			buttons["fireBall"].clicked = 1;
			buttons["snowBall"].clicked = 0;
			buttons["web"].clicked = 0;
			input.available = false;
		}
		else if (isClicked(buttons["snowBall"], input))
		{
			buttons["fireBall"].clicked = 0;
			buttons["snowBall"].clicked = 1;
			buttons["web"].clicked = 0;
			input.available = false;
		}
		else if (isClicked(buttons["web"], input))
		{
			buttons["fireBall"].clicked = 0;
			buttons["snowBall"].clicked = 0;
			buttons["web"].clicked = 1;
			input.available = false;
		}
		else if (isClicked(buttons["train"], input))
		{
			ai.learn(0.5, 0.5);
			input.available = false;
			buttons["train"].clicked = true;
		}
		else if (isClicked(buttons["select"], input))
		{
			ai.save();
			input.available = false;
			buttons["select"].clicked = true;
		}
		else if (fmax(abs(input.birth_x - input.current_x), abs(input.birth_y - input.current_y)) > 0.05 and not(input.birth_x < 0.5 && input.birth_y > 0.5))
		{
			shootingAngle = angleOfVector(input.current_x - input.birth_x, input.current_y - input.birth_y);
			input.available = false;
		}
	}
	if (joystick.clicked)
		player.move(movementAngle);
	if (shootingAngle != INT_MAX)
	{
		if (buttons["fireBall"].clicked)
		{
			player.shoot(shootingAngle, Shooter::fireBall);
			shootedBullet = int(Shooter::fireBall);
		}
		else if (buttons["snowBall"].clicked)
		{
			player.shoot(shootingAngle, Shooter::snowBall);
			shootedBullet = int(Shooter::snowBall);
		}
		else if (buttons["web"].clicked)
		{
			player.shoot(shootingAngle, Shooter::web);
			shootedBullet = int(Shooter::web);
		}
	}
	else
		shootedBullet = -1;
	return true;
}
void InputManager::render(SDL_Renderer *s)
{
	if (joystick.clicked)
	{
		SDL_SetTextureAlphaMod(joystick.padImage, 200);
		SDL_RenderCopy(s, joystick.padImage, NULL, &(joystick.padRect));
		SDL_SetTextureAlphaMod(joystick.buttonImage, 200);
		joystick.buttonRect = createRect((joystick.padRect.x + joystick.padRect.w / 2) + cos(player.get_headingAngle_rad()) * (joystick.padRect.w - joystick.buttonRect.w) / 2, (joystick.padRect.y + joystick.padRect.h / 2) + sin(player.get_headingAngle_rad()) * (joystick.padRect.w - joystick.buttonRect.w) / 2, joystick.buttonRect.w, joystick.buttonRect.h);
		SDL_RenderCopy(s, joystick.buttonImage, NULL, &(joystick.buttonRect));
	}
	else
	{
		SDL_SetTextureAlphaMod(joystick.padImage, 100);
		SDL_RenderCopy(s, joystick.padImage, NULL, &(joystick.padRect));
		SDL_SetTextureAlphaMod(joystick.buttonImage, 100);
		joystick.buttonRect = createRect((joystick.padRect.x + joystick.padRect.w / 2), (joystick.padRect.y + joystick.padRect.h / 2), joystick.buttonRect.w, joystick.buttonRect.h);
		SDL_RenderCopy(s, joystick.buttonImage, NULL, &(joystick.buttonRect));
	}
	for (auto &button : buttons)
	{
		SDL_SetTextureAlphaMod(button.second.image, 100 + button.second.clicked * 155);
		SDL_RenderCopy(s, button.second.image, NULL, &(button.second.rect));
	}
}