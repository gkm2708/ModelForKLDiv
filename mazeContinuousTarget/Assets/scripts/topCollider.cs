using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class topCollider : MonoBehaviour
{

	// Start is called before the first frame update
    void Start()
    {
    }


	public bool state;
	void OnCollisionEnter(Collision col)
	{
		state = true;

	}
    


	// Update is called once per frame
	void Update()
    {
        
    }
}
