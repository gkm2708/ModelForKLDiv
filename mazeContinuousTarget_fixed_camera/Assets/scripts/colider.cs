using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class colider : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        
    }


	public bool stat;
	void OnCollisionEnter(Collision col)
	{
		stat = true;

	}


    // Update is called once per frame
    void Update()
    {
        
    }
}
