#ifndef shooter_cxx
#define shooter_cxx
class Shooter
{
	friend class Brain;
	friend void collisionHandeler(Shooter &, Shooter &);
	friend void animateDeath(Shooter &, SDL_Renderer *, std::unordered_map<std::string, SDL_Texture *>);

  public:
	enum bulletType
	{
		snowBall,
		fireBall,
		web
	};
	enum flag
	{
		none,
		shooter_frozen,
		shooter_jammed,
		shooter_dead
	};
	struct Bullet
	{
		enum flag
		{
			available,
			travelling,
			onCoolDown
		};
		static constexpr int RADIUS = 30, SPEED = 10, COOL_DOWN_TIME = 100, FREEZ_PERSISTANCE = 100, JAM_PERSISTANCE = 150, FIRE_BALL_EFFECT = 1, FREEZING_EFFECT = 1;
		float x, y;
		unsigned int shootingTime;
		enum flag state = Bullet::available;
		enum bulletType type;
		float shootingAngle_degree;
	};

  private:
	//**** Constants ****
	static constexpr int RADIUS = 50, GUN_RADIUS = 20, SPEED = 7, RECOIL = 10; // speed is in ∆pixels/frame;
	int SCREEN_WIDTH, SCREEN_HEIGHT;
	std::unordered_map<std::string, SDL_Texture *> images;
	//**** Variables****
	float x, y;
	int gunAngle_degree = 0, headingAngle_degree = 0;
	unsigned int time = 0, health = 0, lastDamageTime = 0, statusEffectRemaining = 0;
	struct Bullet lastDamageBullet;
	enum flag state = shooter_dead;
	std::vector<struct Bullet> magazine;
	int availableBulletCount;
	ParticleEffect sprinkler;
	std::vector<SDL_Texture *> snowParticleFrames, fireParticleFrames, magicParticleFrames, snowBurstFrames, fireBurstFrames, sparkParticleFrames;

	inline unsigned int deltaTime(unsigned int passedTime)
	{
		return time - passedTime;
	}
	void shift(int dx, int dy);
	void inflictDamage(struct Bullet);
	inline void rotateTo(int angle);
	void sprinkleSnow(int x, int y, int particleCount, int sprinkleRandomness, int movementAngle_degree = 0, int speed = 0);
	void sprinkleFire(int x, int y, int particleCount, int sprinkleRandomness, int movementAngle_degree, int speed = 5);
	void animateMoveMagic(int particleCount);
	void animateMagic();

  public:
	Shooter() {}
	Shooter(int arg_SCREEN_WIDTH, int arg_SCREEN_HEIGHT, int arg_x, int arg_y, int magazine_size, std::unordered_map<std::string, SDL_Texture *> arg_images) : SCREEN_WIDTH(arg_SCREEN_WIDTH), SCREEN_HEIGHT(arg_SCREEN_HEIGHT), images(arg_images), x(arg_x), y(arg_y), health(10), state(none), magazine(magazine_size)
	{
		availableBulletCount = magazine.size();
		snowParticleFrames.emplace_back(images["snowParticle_1"]);
		snowParticleFrames.emplace_back(images["snowParticle_2"]);
		fireParticleFrames.emplace_back(images["fireParticle_1"]);
		fireParticleFrames.emplace_back(images["fireParticle_2"]);
		magicParticleFrames.emplace_back(images["magic_1"]);
		magicParticleFrames.emplace_back(images["magic_2"]);
		magicParticleFrames.emplace_back(images["magic_3"]);
		snowBurstFrames.emplace_back(images["snowBurst"]);
		fireBurstFrames.emplace_back(images["fireBurst"]);
		sparkParticleFrames.emplace_back(images["spark_1"]);
		sparkParticleFrames.emplace_back(images["spark_2"]);
	}
	inline bool isAlive();
	inline void respawn(std::unordered_map<std::string, SDL_Texture *> images);
	inline int get_availableBulletCount();
	inline float get_headingAngle_rad();
	void render(SDL_Renderer *);
	void move(int angle_degree);
	bool shoot(float shootingAngle_degree, enum bulletType);
	void update();
};

void Shooter::shift(int dx, int dy)
{
	x += dx;
	y += dy;
	x += (SCREEN_WIDTH - RADIUS - x) * (x > SCREEN_WIDTH - RADIUS) + (RADIUS - x) * (x < RADIUS);
	/*if (x > SCREEN_WIDTH - RADIUS)
		x = SCREEN_WIDTH - RADIUS;
	else if (x < RADIUS)
		x = RADIUS;*/
	y += (SCREEN_HEIGHT - RADIUS - y) * (y > SCREEN_HEIGHT - RADIUS) + (RADIUS - y) * (y < RADIUS);
	/*if (y > SCREEN_HEIGHT - RADIUS)
		y = SCREEN_HEIGHT - RADIUS;
	else if (y < RADIUS)
		y = RADIUS;*/
}
void Shooter::inflictDamage(struct Shooter::Bullet bullet)
{
	if (not isAlive())
		return;
	if (bullet.type == fireBall)
		health -= Bullet::FIRE_BALL_EFFECT;
	lastDamageTime = time;
	switch (bullet.type)
	{
	case snowBall:
		state = shooter_frozen;
		statusEffectRemaining = Bullet::FREEZ_PERSISTANCE;
		shift(RECOIL * cos(bullet.shootingAngle_degree * M_PI / 180),
			  RECOIL * sin(bullet.shootingAngle_degree * M_PI / 180));
		sprinkleSnow(bullet.x, bullet.y, 20, Bullet::RADIUS, bullet.shootingAngle_degree, 5);
		sprinkler.addParticle(time, 5, bullet.x, bullet.y, 0, Bullet::RADIUS,
							  255, 150, 0, 0,
							  snowBurstFrames);
		break;
	case fireBall:
		shift(RECOIL * cos(bullet.shootingAngle_degree * M_PI / 180),
			  RECOIL * sin(bullet.shootingAngle_degree * M_PI / 180));
		sprinkleFire(bullet.x, bullet.y, 20, Bullet::RADIUS, bullet.shootingAngle_degree, 5);
		sprinkler.addParticle(time, 5, bullet.x, bullet.y, 0, Bullet::RADIUS,
							  255, 150, 0, 0,
							  fireBurstFrames);
		break;
	case web:
		state = shooter_jammed;
		statusEffectRemaining = Bullet::JAM_PERSISTANCE;
	}
	lastDamageBullet = bullet;
}
inline void Shooter::rotateTo(int angle)
{
	gunAngle_degree += angle - headingAngle_degree;
	gunAngle_degree += -360 * (gunAngle_degree > 360) + 360 * (gunAngle_degree < -360); //keeping angle between -360 to 360 to avoid int overflow in long run
	headingAngle_degree = angle;
}
void Shooter::sprinkleSnow(int x, int y, int particleCount, int sprinkleRandomness, int movementAngle_degree, int speed)
{
	static constexpr int lifeTime = 20, startRadius = 10, endRadius = 5, trailingOpacity = 150;
	sprinkleRandomness++;
	while (particleCount--)
	{
		int rand_x = x + rand() % (2 * sprinkleRandomness) - sprinkleRandomness;
		int rand_y = y + rand() % (2 * sprinkleRandomness) - sprinkleRandomness;
		float opacityFactor = 1 - float(abs(rand_x - x) + abs(rand_y - y) + 1) / (2 * sprinkleRandomness + 1);
		sprinkler.addParticle(time, lifeTime, rand_x, rand_y, startRadius, endRadius,
							  trailingOpacity + int((255 - trailingOpacity) * opacityFactor), 0, speed, movementAngle_degree,
							  snowParticleFrames);
	}
}
void Shooter::sprinkleFire(int x, int y, int particleCount, int sprinkleRandomness, int movementAngle_degree, int speed)
{
	static constexpr int lifeTime = 10, startRadius = 15;
	sprinkleRandomness++;
	while (particleCount--)
	{
		int rand_x = x + rand() % (2 * sprinkleRandomness) - sprinkleRandomness;
		int rand_y = y + rand() % (2 * sprinkleRandomness) - sprinkleRandomness;
		float radiusFactor = 1 - float(abs(rand_x - x) + abs(rand_y - y) + 1) / (2 * sprinkleRandomness + 1);
		sprinkler.addParticle(time, lifeTime, rand_x, rand_y, startRadius * radiusFactor, 0,
							  255, 150, speed, movementAngle_degree,
							  fireParticleFrames, 1);
	}
}
void Shooter::animateMoveMagic(int particleCount)
{
	static constexpr int lifeTime = 20, startRadius = 15, sprinkleRandomness_degree = 60 + 1, speed = 3;
	while (particleCount--)
	{
		int randAngle_degree = headingAngle_degree + 180 + rand() % (2 * sprinkleRandomness_degree) - sprinkleRandomness_degree;
		int rand_x = x + RADIUS * cos(randAngle_degree * M_PI / 180);
		int rand_y = y + RADIUS * sin(randAngle_degree * M_PI / 180);
		sprinkler.addParticle(time, lifeTime, rand_x, rand_y, startRadius, 0,
							  255, 150, speed, headingAngle_degree + 180,
							  sparkParticleFrames);
	}
}
void Shooter::animateMagic()
{
	static constexpr int lifeTime = 2, radius = 20, bulletSprinkleAntiFactor = 5;
	if (not isAlive())
		return;
	int magic_x = x + (GUN_RADIUS + radius) * cos(gunAngle_degree * M_PI / 180),
		magic_y = y + (GUN_RADIUS + radius) * sin(gunAngle_degree * M_PI / 180);
	if (get_availableBulletCount() and not(state == shooter_jammed && statusEffectRemaining) and isAlive())
	{
		sprinkleSnow(magic_x, magic_y, 1 + get_availableBulletCount() / bulletSprinkleAntiFactor, 20);
		sprinkleFire(magic_x, magic_y, 1 + get_availableBulletCount() / bulletSprinkleAntiFactor, 20, 0, 0);
	}
	sprinkler.addParticle(time, lifeTime, magic_x, magic_y, radius, radius,
						  255, 150, 0, 0,
						  magicParticleFrames, 1);
}
inline bool Shooter::isAlive()
{
	return state != shooter_dead;
}
inline void Shooter::respawn(std::unordered_map<std::string, SDL_Texture *> images)
{
	this->images["gun"] = images["gun"];
	health = 10;
	state = none;
}
inline int Shooter::get_availableBulletCount()
{
	return availableBulletCount;
}
inline float Shooter::get_headingAngle_rad()
{
	return headingAngle_degree * M_PI / 180;
}
void Shooter::render(SDL_Renderer *s)
{
	SDL_Rect temp_rect = createRect(x, y, 2 * RADIUS, 2 * RADIUS);
	SDL_RenderCopyEx(s, images["shooter"], NULL, &temp_rect, headingAngle_degree, NULL, SDL_FLIP_NONE);
	SDL_SetTextureAlphaMod(images["frost"], 255 * float((state == shooter_frozen) * statusEffectRemaining) / Bullet::FREEZ_PERSISTANCE);
	SDL_RenderCopyEx(s, images["frost"], NULL, &temp_rect, headingAngle_degree, NULL, SDL_FLIP_NONE);
	animateMagic();
	for (struct Bullet &bullet : magazine)
	{
		if (bullet.state != Bullet::travelling)
			continue;
		temp_rect = createRect(bullet.x, bullet.y, Bullet::RADIUS, Bullet::RADIUS);
		switch (bullet.type)
		{
		case snowBall:
			SDL_RenderCopy(s, images["snowBall"], NULL, &temp_rect);
			sprinkleSnow(bullet.x, bullet.y, 1, 10);
			break;
		case fireBall:
			SDL_RenderCopy(s, images["fireBall"], NULL, &temp_rect);
			sprinkleFire(bullet.x, bullet.y, 4, 10, bullet.shootingAngle_degree + 180);
			break;
		case web:
			SDL_RenderCopy(s, images["web"], NULL, &temp_rect);
		}
	}
	sprinkler.render(s, time);
	temp_rect = createRect(x, y, 2 * GUN_RADIUS, 2 * GUN_RADIUS);
	SDL_RenderCopyEx(s, images["gun"], NULL, &temp_rect, gunAngle_degree, NULL, SDL_FLIP_NONE);
}
void Shooter::move(int angle_degree)
{
	if (not isAlive())
		return;
	rotateTo(angle_degree);
	float speed_factor;
	switch (state)
	{
	case shooter_frozen:
		speed_factor = (Bullet::FREEZ_PERSISTANCE - statusEffectRemaining) / float(Bullet::FREEZ_PERSISTANCE * Bullet::FREEZING_EFFECT);
		break;
	default:
		speed_factor = 1;
	}
	shift(cos((M_PI / 180) * angle_degree) * SPEED * speed_factor,
		  sin((M_PI / 180) * angle_degree) * SPEED * speed_factor);
	animateMoveMagic(2 * speed_factor);
}
bool Shooter::shoot(float shootingAngle_degree, enum bulletType type)
{
	if (not isAlive() or (state == shooter_jammed && statusEffectRemaining))
		return false;
	for (struct Bullet &bullet : magazine)
	{
		if (bullet.state == Bullet::available)
		{
			bullet.state = Bullet::travelling;
			bullet.shootingTime = time;
			bullet.x = x;
			bullet.y = y;
			bullet.shootingAngle_degree = shootingAngle_degree;
			gunAngle_degree = shootingAngle_degree;
			bullet.type = type;
			if (type == snowBall or type == fireBall)
			{
				shift(-RECOIL * cos(shootingAngle_degree * M_PI / 180),
					  -RECOIL * sin(shootingAngle_degree * M_PI / 180));
			}
			return true;
		}
	}
	return false;
}
void Shooter::update()
{
	if (health <= 0)
		state = shooter_dead;
	availableBulletCount = 0;
	for (struct Bullet &bullet : magazine)
	{
		if (bullet.state == Bullet::travelling)
		{
			bullet.x += Bullet::SPEED * cos(bullet.shootingAngle_degree * M_PI / 180);
			bullet.y += Bullet::SPEED * sin(bullet.shootingAngle_degree * M_PI / 180);
			if (bullet.y > SCREEN_HEIGHT or bullet.y < 0 or bullet.x > SCREEN_WIDTH or bullet.x < 0)
				bullet.state = Bullet::onCoolDown;
		}

		if (bullet.state == Bullet::onCoolDown and deltaTime(bullet.shootingTime) > Bullet::COOL_DOWN_TIME)
			bullet.state = Bullet::available;
		availableBulletCount += (bullet.state == Bullet::available);
	}
	time++;
	statusEffectRemaining -= (statusEffectRemaining != 0);
}

void collisionHandeler(Shooter &player1, Shooter &player2)
{
	// **** PLAYER TO BULLET COLLISION CHECK ****
	for (struct Shooter::Bullet &bullet : player2.magazine)
	{
		if (bullet.state == Shooter::Bullet::travelling and sqrt((bullet.x - player1.x) * (bullet.x - player1.x) + (bullet.y - player1.y) * (bullet.y - player1.y)) < Shooter::Bullet::RADIUS + Shooter::RADIUS)
		{
			player1.inflictDamage(bullet);
			bullet.state = Shooter::Bullet::onCoolDown;
		}
	}
	for (struct Shooter::Bullet &bullet : player1.magazine)
	{
		if (bullet.state == Shooter::Bullet::travelling and sqrt((bullet.x - player2.x) * (bullet.x - player2.x) + (bullet.y - player2.y) * (bullet.y - player2.y)) < Shooter::Bullet::RADIUS + Shooter::RADIUS)
		{
			player2.inflictDamage(bullet);
			bullet.state = Shooter::Bullet::onCoolDown;
		}
	}
	// **** PLAYER TO PLAYER COLLISION CHECK ****
	int intersectingDistance = 2 * Shooter::RADIUS - sqrt((player1.x - player2.x) * (player1.x - player2.x) + (player1.y - player2.y) * (player1.y - player2.y));
	if (intersectingDistance > 0)
	{
		int angle_player1ToPlayer2 = angleOfVector(player2.x - player1.x, player2.y - player1.y);
		float normal_x = intersectingDistance * cos(angle_player1ToPlayer2 * M_PI / 180) / 2,
			  normal_y = intersectingDistance * sin(angle_player1ToPlayer2 * M_PI / 180) / 2;
		player1.shift(-normal_x, -normal_y);
		player2.shift(normal_x, normal_y);
	}
}
#endif