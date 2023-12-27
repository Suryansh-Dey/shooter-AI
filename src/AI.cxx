#include "NeuralNetwork2.cxx"
class Database
{
	friend class Brain;
	struct Event
	{
		double x, y, state, opponent_x, opponent_y, opponentState, bulletsLeft, movementAngle_degree, bulletType, bulletAngle_degree;
		std::vector<Shooter::Bullet> bullets;
	};
	std::vector<struct Event> events;
	std::vector<double> rewards;

  public:
	inline int getSize();
	void insert(double x, double y, double state, double opponent_x, double opponent_y, double opponentState, double bulletsLeft, std::vector<Shooter::Bullet> bullets, double movementAngle_degree, double bulletType, double bulletAngle_degree);
	void clear();
	inline bool assignReward(double reward);
};
inline int Database::getSize()
{
	return events.size();
}
void Database::insert(double x, double y, double state, double opponent_x, double opponent_y, double opponentState, double bulletsLeft, std::vector<Shooter::Bullet> bullets, double movementAngle_degree, double bulletType, double bulletAngle_degree)
{
	events.emplace_back(Event{x, y, state, opponent_x, opponent_y, opponentState, bulletsLeft, movementAngle_degree, bulletType, bulletAngle_degree, bullets});
}
void Database::clear()
{
	events.resize(0);
	rewards.resize(0);
}
inline bool Database::assignReward(double reward)
{
	if (events.size() <= rewards.size())
		return 0;
	rewards.emplace_back(reward);
	return 1;
}
class Brain
{
	NN model, model2;
	Database data;
	int magazineSize, SCREEN_WIDTH, SCREEN_HEIGHT;

  public:
	Brain(int magazineSize, int SCREEN_WIDTH, int SCREEN_HEIGHT) : magazineSize(magazineSize), SCREEN_WIDTH(SCREEN_WIDTH), SCREEN_HEIGHT(SCREEN_HEIGHT)
	{
		std::vector<int> neuron_cts{4 * int(magazineSize) + 7, 4, 4, 3}; // we will input x,y of every bullet of opponent and their type and angle of movement. Also the x,y of opponent, of it's own, opponent's state (freezed or jammed), it's own state and number of bullets left.
		std::ifstream file("agent.txt");
		if (file.good())
		{
			model = NN_load("agent.txt");
			model2 = NN_load("agent.txt");
			NN_set_activationFunction(model, 4, NN_type_Linear);
			NN_set_activationFunction(model2, 4, NN_type_Linear);
		}
		else
		{
			model = NN_create(neuron_cts);
			NN_set_activationFunction(model, 4, NN_type_Linear);
			model2 = NN_create(neuron_cts);
			NN_set_activationFunction(model2, 4, NN_type_Linear);
			NN_clear(model);
			NN_mutate(model, 1, 0.5);
			NN_copy(model2, model);
		}
		NN_set_learningRate(0.1);
		NN_set_maxWeight(5);
	}
	~Brain()
	{
		//NN_show(model);
	}
	//void memorize(Shooter &player1, Shooter &player2, double shootedBullet);
	void learn(double probability, double change);
	void invokeAI(Shooter &player1, Shooter &player2);
	void save();
};
/*void Brain::memorize(Shooter &player1, Shooter &player2, double shootedBullet)
{
	data.insert(player1.x, player1.y, player1.state, player2.x, player2.y, player2.state, player1.get_availableBulletCount(), player2.magazine, player1.headingAngle_degree, shootedBullet, player1.gunAngle_degree);
	data.assignReward(1);
}*/
void Brain::learn(double probability, double change)
{
	NN_copy(model, model2);
	NN_mutate(model, probability, change);
	/*
	if (data.rewards.size() != data.events.size())
	{
		printf("\nERROR: Database doesn't has equal number of events and rewards. Note: number of events = %i and rewards = %i\nPolicyAi::learnByPolicyGradient() Aborted", int(data.events.size()), int(data.rewards.size()));
		exit(0);
	}
	NN_trainingData examples(model, data.getSize());
	for (const auto &event : data.events)
	{
		std::vector<double> inputs = std::vector<double>{event.x / SCREEN_WIDTH, event.y / SCREEN_HEIGHT, event.state, event.opponent_x / SCREEN_WIDTH, event.opponent_y / SCREEN_HEIGHT, event.opponentState, event.bulletsLeft / magazineSize};
		for (int magazineNo = 0; magazineNo < magazineSize; magazineNo++)
		{
			if (event.bullets[magazineNo].state != Shooter::Bullet::travelling)
			{
				inputs.emplace_back(-1);
				inputs.emplace_back(-1);
				inputs.emplace_back(-1);
				inputs.emplace_back(-1);
			}
			else
			{
				inputs.emplace_back(event.bullets[magazineNo].x / double(SCREEN_WIDTH));
				inputs.emplace_back(event.bullets[magazineNo].y / double(SCREEN_HEIGHT));
				inputs.emplace_back(event.bullets[magazineNo].shootingAngle_degree / 180.0f);
				inputs.emplace_back(event.bullets[magazineNo].type);
			}
		}
		examples.insert(inputs, std::vector<double>{event.movementAngle_degree / 360.0f, event.bulletType, event.bulletAngle_degree / 180.0f});
		NN_input(model, inputs);std::cout<<NN_vectorOutput(NN_process(model))[0]<<',';
		std::cout<<event.movementAngle_degree / 360.0f<<'\n';
	}
	std::cout<<'\n'<<NN_totalCost(model, examples)<<','<<examples.get_size()<<'\n';
	for (int stepNo = 0; stepNo < steps; stepNo++)
		NN_decendGradient(model, examples);
	data.clear();
	*/
}
void Brain::invokeAI(Shooter &player1, Shooter &player2)
{
	auto inputs = std::vector<double>{player1.x, player1.y , double(player1.state), player2.x , player2.y , double(player2.state), double(player1.get_availableBulletCount())};
	for (int bulletNo = 0; bulletNo < magazineSize; bulletNo++)
	{
		if (player2.magazine[bulletNo].state != Shooter::Bullet::travelling)
		{
			inputs.emplace_back(-1);
			inputs.emplace_back(-1);
			inputs.emplace_back(-1);
			inputs.emplace_back(-1);
		}
		else
		{
			inputs.emplace_back(player2.magazine[bulletNo].x);
			inputs.emplace_back(player2.magazine[bulletNo].y);
			inputs.emplace_back(player2.magazine[bulletNo].shootingAngle_degree);
			inputs.emplace_back(player2.magazine[bulletNo].type);
		}
	}
	NN_input(model, inputs);
	std::vector<double> output = NN_vectorOutput(NN_process(model));
	player1.move(output[0]);
	if (output[1] >= 0 and output[1]<30)
		player1.shoot(angleOfVector(player2.x-player1.x,player2.y-player1.y), (Shooter::bulletType)(output[1]/10));//output[2]
}
void Brain::save()
{
	NN_copy(model2, model);
	NN_save(model, "agent.txt");
	std::cout<<"saved!\n";
}