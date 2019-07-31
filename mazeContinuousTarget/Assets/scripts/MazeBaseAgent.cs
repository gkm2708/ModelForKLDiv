using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class MazeBaseAgent : Agent
{
    // Start is called before the first frame update

	public GameObject Ball;
	public Transform Target;

	private GameObject wall;
	private GameObject temp; 

	private colider script; // this will be the container of the script

	void Start()
    {
    }

	public override void AgentReset(){
		this.transform.position = new Vector3 (0, 0, 0);
		this.transform.rotation = new Quaternion (0, 0, 0, 1);

		Target.position = new Vector3 (UnityEngine.Random.value*3.5f - 1.75f, 0.05f, UnityEngine.Random.value*3.5f - 1.75f);
		//Target.position = new Vector3 (0.0f, 0.05f, 0.0f);
		//Ball.transform.position = new Vector3 (UnityEngine.Random.value * 4.0f - 2.0f, 0.30f, UnityEngine.Random.value * 4.0f - 2.0f);
		Ball.transform.position = new Vector3 (0.0f, 0.05f, 0.0f);
	}


	public override void CollectObservations(){

		AddVectorObs (gameObject.transform.rotation.x);
		AddVectorObs (gameObject.transform.rotation.z);

		AddVectorObs (Ball.transform.position.x);
		AddVectorObs (Ball.transform.position.z);

		AddVectorObs (Target.position.x);
		AddVectorObs (Target.position.z);
	}

	public float speed = 0.1f;

	public override void AgentAction(float[] vectorAction, string textAction){

		Vector3 BallTransform = Ball.transform.position;

		var actionZ = 2f * Mathf.Clamp(vectorAction[0], -1f, 1f);
		var actionX = 2f * Mathf.Clamp(vectorAction[1], -1f, 1f);

		if ((gameObject.transform.rotation.z < 0.25f && actionZ > 0f) ||
			(gameObject.transform.rotation.z > -0.25f && actionZ < 0f)) {

			gameObject.transform.Rotate(new Vector3(0, 0, 1), actionZ);
		}

		if ((gameObject.transform.rotation.x < 0.25f && actionX > 0f) ||
			(gameObject.transform.rotation.x > -0.25f && actionX < 0f)) {

			gameObject.transform.Rotate(new Vector3(1, 0, 0), actionX);
		}
			
		float distanceToTarget = Vector3.Distance (Ball.transform.position, Target.position);

		if (distanceToTarget < 0.5f) {
			SetReward (0.0f);
			Done ();
		} else if (BallTransform.y - gameObject.transform.position.y < -2f ||
		           Mathf.Abs (BallTransform.x - gameObject.transform.position.x) > 2.5f ||
		           Mathf.Abs (BallTransform.z - gameObject.transform.position.z) > 2.5f) {
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
	}
}
