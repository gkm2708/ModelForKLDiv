using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class MazeBaseAgent : Agent
{
    // Start is called before the first frame update

	public GameObject Ball;
	public GameObject Goal;
	//public Transform Target;

	public int[,] Maze;

	public float y_pos_ball = 0.75f;
	public float y_pos_goal = 0.51f;

	public float delta_rot_x;
	public float delta_rot_z;

	private GameObject wall;
	private GameObject temp; 

	private colider script; // this will be the container of the script

	void Start()
    {

		Maze = new int[,]{{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};


		/*

		// first create the base board

		int dim = Maze.Rank;
		int row = Maze.GetLength(0);
		int columns = Maze.GetLength(1);


		GameObject prefab = Resources.Load("Cube") as GameObject;
		for (int i=0; i<row; i++){
			for (int j = 0; j < columns; j++) {

				// check if array has 0 at a place
				if (Maze[i,j] == 0){
					GameObject go = Instantiate (prefab) as GameObject;
					go.transform.position = new Vector3 (-33.0f +i , 0.55f, -33.0f + j);
					go.transform.parent = gameObject.transform;
				}
			}
		}

		*/
		/*
		int ball_x = 0;
		int ball_y = 0;

		// Random ball position in feasible space
		ball_x = UnityEngine.Random.Range(1, 67);
		ball_y = UnityEngine.Random.Range(1, 67);

		while (Maze [ball_x, ball_y] == 0) {
			ball_x = UnityEngine.Random.Range(1, 67);
			ball_y = UnityEngine.Random.Range(1, 67);
		}
			
		Ball.transform.position = new Vector3 (-33.0f + ball_x, y_pos, -33.0f + ball_y);

		// Random ball position in feasible space
		ball_x = UnityEngine.Random.Range(1, 67);
		ball_y = UnityEngine.Random.Range(1, 67);

		while (Maze [ball_x, ball_y] == 0) {
			ball_x = UnityEngine.Random.Range(1, 67);
			ball_y = UnityEngine.Random.Range(1, 67);
		}
			

		Goal.transform.position = new Vector3 (-33.0f + ball_x , y_pos, -33.0f + ball_y);
		//Goal.transform.parent = gameObject.transform;
				 
		Ball.GetComponent<Rigidbody>().AddForce(0, 0, 0);
		Ball.GetComponent<Rigidbody>().velocity = new Vector3(0.0f, 0.0f, 0.0f);
		*/
		//Ball.transform.position = new Vector3 (0.0f, 0.55f, 0.0f);
		//Target.position = new Vector3 (0.0f, 0.05f, 0.0f);
    }







	public override void AgentReset(){
		
		//Ball.transform.position = new Vector3 (0.0f, 0.55f, 0.0f);
		//Target.position = new Vector3 (0.0f, 0.05f, 0.0f);

		int ball_x = 0;
		int ball_y = 0;

		// Random ball position in feasible space
		ball_x = UnityEngine.Random.Range(1, 67);
		ball_y = UnityEngine.Random.Range(1, 67);

		while (Maze [ball_x, ball_y] == 0) {
			ball_x = UnityEngine.Random.Range(1, 67);
			ball_y = UnityEngine.Random.Range(1, 67);
		}

		Ball.transform.position = new Vector3 (-33.0f + ball_x, y_pos_ball, -33.0f + ball_y);

		// Random ball position in feasible space
		ball_x = UnityEngine.Random.Range(1, 67);
		ball_y = UnityEngine.Random.Range(1, 67);

		while (Maze[ball_x, ball_y] == 0) {
			ball_x = UnityEngine.Random.Range(1, 67);
			ball_y = UnityEngine.Random.Range(1, 67);
		}

		Goal.transform.position = new Vector3 (-33.0f + ball_x, y_pos_goal, -33.0f + ball_y);
		Goal.transform.parent = gameObject.transform;

		Ball.GetComponent<Rigidbody>().AddForce(0, 0, 0);
		Ball.GetComponent<Rigidbody>().velocity = new Vector3(0.0f, 0.0f, 0.0f);

		gameObject.transform.position = new Vector3 (0, 0, 0);
		gameObject.transform.rotation = new Quaternion (0.0f, 0.0f, 0.0f, 1);

	}








	public override void CollectObservations(){

		AddVectorObs (gameObject.transform.rotation.x);
		AddVectorObs (gameObject.transform.rotation.z);

		AddVectorObs (Ball.transform.position.x);
		AddVectorObs (Ball.transform.position.z);

		AddVectorObs (Goal.transform.position.x);
		AddVectorObs (Goal.transform.position.z);

		AddVectorObs (Ball.GetComponent<Rigidbody>().velocity.x);
		AddVectorObs (Ball.GetComponent<Rigidbody>().velocity.z);

		//Debug.Log (Ball.GetComponent<Rigidbody> ().velocity.x);
	}


	void OnGUI() {
		GUI.Label (new Rect (200, 8, 240, 160), "Angular Rotation: " + gameObject.transform.eulerAngles);
		GUI.Label (new Rect (200, 40, 240, 160), "Angular Velocity: " + gameObject.GetComponent<Rigidbody>().angularVelocity);
		GUI.Label (new Rect (200, 72, 240, 160), "Delta Rotation: " + delta_rot_x +" "+ delta_rot_z);
	}

	/*

	public override void AgentAction(float[] vectorAction, string textAction){

		//Vector3 BallTransform = Ball.transform.position;
		//Ball.GetComponent<Rigidbody>().AddForce(vectorAction[2], 0, vectorAction[3]);
		//Ball.GetComponent<Rigidbody>().velocity = new Vector3(vectorAction[2], 0.0f, vectorAction[3]);

		//Debug.Log (actionZ);
		//Debug.Log (actionX);


		// desired rotation
		var actionZ = vectorAction[0];
		var actionX = vectorAction[1];


		// current rotation
		float current_rot_x = gameObject.transform.rotation.x;
		float current_rot_z = gameObject.transform.rotation.z;


		// delta rotation
		delta_rot_x = actionX - current_rot_x;
		delta_rot_z = actionZ - current_rot_z;



		gameObject.GetComponent<Rigidbody>().angularVelocity = new Vector3(delta_rot_x/10, 0.0f, delta_rot_z/10);

		//var actionZ = 2f * Mathf.Clamp(vectorAction[0], -1f, 1f);
		//var actionX = 2f * Mathf.Clamp(vectorAction[1], -1f, 1f);

		if ((gameObject.transform.rotation.x < 0.25f && actionX > 0f) ||
			(gameObject.transform.rotation.x > -0.25f && actionX < 0f)) {

			//gameObject.GetComponent<Rigidbody>().angularVelocity = new Vector3(delta_rot_x/10, 0.0f, 0.0f);
			gameObject.transform.Rotate(new Vector3(1, 0, 0), actionX);
		}

		if ((gameObject.transform.rotation.z < 0.25f && actionZ > 0f) ||
			(gameObject.transform.rotation.z > -0.25f && actionZ < 0f)) {

			gameObject.transform.Rotate(new Vector3(0, 0, 1), actionZ);
		}

		//Debug.Log (gameObject.transform.rotation);

		// moving ball to next position
		int x_pos = (int)(-36.0f + vectorAction[0]); 
		int y_pos = (int)(36.0f - vectorAction[1]);

		Debug.Log (x_pos);
		Debug.Log (y_pos);

		SetReward (0.0f);

		if (Maze [(int)vectorAction[0], (int)vectorAction[1]] == 1) {
			Ball.transform.position = new Vector3 (x_pos, 0.55f, y_pos);
			SetReward (0.1f);
	}
	*/

	public override void AgentAction(float[] vectorAction, string textAction){

		Vector3 BallTransform = Ball.transform.position;

		//var actionZ = 2f * Mathf.Clamp(vectorAction[0], -1f, 1f);
		//var actionX = 2f * Mathf.Clamp(vectorAction[1], -1f, 1f);

		// desired rotation
		var actionZ = vectorAction[0];
		var actionX = vectorAction[1];

		// current rotation
		float current_rot_x = gameObject.transform.eulerAngles.x;
		float current_rot_z = gameObject.transform.eulerAngles.z;

		// delta rotation
		delta_rot_x = actionX - current_rot_x;
		delta_rot_z = actionZ - current_rot_z;


		//gameObject.GetComponent<Rigidbody>().angularVelocity = new Vector3((float)delta_rot_x/10, 0.0f, (float)delta_rot_z/10);


		if ((-5.25f < gameObject.transform.eulerAngles.z && 
			gameObject.transform.eulerAngles.z < 5.25f)) {

			gameObject.transform.Rotate(new Vector3(0, 0, 1), actionZ);

		} else if ((-5.25f > gameObject.transform.eulerAngles.z && actionZ > 0.0f) ||
			(5.25f < gameObject.transform.eulerAngles.z && actionZ < 0.0f)) {

			gameObject.transform.Rotate(new Vector3(0, 0, 1), actionZ);
		
		}


		if ((-5.25f < gameObject.transform.eulerAngles.x && 
			gameObject.transform.eulerAngles.x < 5.25f)) {

			gameObject.transform.Rotate(new Vector3(1, 0, 0), actionX);

		} else if ((-5.25f > gameObject.transform.eulerAngles.x && actionX > 0.0f) ||
			(5.25f < gameObject.transform.eulerAngles.x && actionX < 0.0f)) {

			gameObject.transform.Rotate(new Vector3(1, 0, 0), actionX);

		}


		float distanceToTarget = Vector3.Distance (Ball.transform.position, Goal.transform.position);

		if (distanceToTarget < 0.5f) {
			SetReward (0.0f);
			Done ();
		} else if (BallTransform.y - gameObject.transform.position.y < -20f ||
			Mathf.Abs (BallTransform.x - gameObject.transform.position.x) > 33.5f ||
			Mathf.Abs (BallTransform.z - gameObject.transform.position.z) > 33.5f) {
			SetReward (-11.0f);
			Done ();
		}



		/*

		if ((0.0f < gameObject.transform.rotation.x
			&& gameObject.transform.rotation.x < 5.0f 
			&& actionX > 0.0f) ||
			(0.0f > gameObject.transform.rotation.x
				&& gameObject.transform.rotation.x > -5.0f 
				&& actionX < 0.0f)) {

			gameObject.transform.Rotate(new Vector3(1, 0, 0), actionX);
		}

		if (distanceToTarget < 0.5f) {
			SetReward (0.0f);
			Done ();
		} else if (BallTransform.y - gameObject.transform.position.y < -20f ||
			Mathf.Abs (BallTransform.x - gameObject.transform.position.x) > 33.5f ||
			Mathf.Abs (BallTransform.z - gameObject.transform.position.z) > 33.5f) {
			SetReward (-11.0f);
			Done ();
		} else {
			bool found = false;
			for (int i = 0; i < this.transform.childCount; i++) {
				if (this.transform.GetChild (i).gameObject.tag == "wall") {
					script = this.transform.GetChild (i).gameObject.GetComponent<colider> ();
					if (script.stat == true) {
						SetReward (-11.0f);
						found = true;
						script.stat = false;
					}
				}
			}
			if (found == false) {
				SetReward (-10.0f);
			}
		}


		*/

	}
}
