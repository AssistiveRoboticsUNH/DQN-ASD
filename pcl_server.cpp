#include "ros/ros.h"
#include "std_srvs/Empty.h"
#include "deep_q_network/PointCloud2Array.h"

#include <pcl/ros/conversions.h> 
#include <pcl_ros/point_cloud.h>

bool batchConvertPC(deep_q_network::PointCloud2Array::Request &req,
			deep_q_network::PointCloud2Array::Response &res)
{
  	ROS_INFO("request made");

  	for(int i = 0; i < req.num_seq; i++){
		pcl::PointCloud<pcl::PointXYZ> input_;
		pcl::fromROSMsg(req.msg_array[i], input_);

		unsigned height = req.msg_array[i].height;
		unsigned width = req.msg_array[i].width;
		double distance_factor = 1;//8.5

		for (unsigned y = 0; y < height; ++y){
		   	for ( unsigned x = 0; x < width; ++x){
		   		
		   		short val = input_[y*(width)+x].z * 100 * distance_factor;
		   		if(val > 255)
		   			val = 255;
		   		/*if(val > 3440)
		   			val = 3440;*/
		   		res.points.push_back(val);

		    }
		}

	}

  	return true;
}
/*
bool singleConvertPC(deep_q_network::PointCloud2Array::Request &req,
			deep_q_network::PointCloud2Array::Response &res)
{
  	ROS_INFO("request made");

  	for(int i = 0; i < req.num_seq; i++){
		pcl::PointCloud<pcl::PointXYZ> input_;
		pcl::fromROSMsg(req.msg_array[i], input_);

		unsigned height = req.msg_array[i].height;
		unsigned width = req.msg_array[i].width;
		double distance_factor = 8.5;

		for (unsigned y = 0; y < height; ++y){
		   	for ( unsigned x = 0; x < width; ++x){
		   		
		   		short val = input_[y*(width)+x].z * 100 * distance_factor;
		   		if(val > 3440)
		   			val = 3440;
		   		res.points.push_back(val);

		    }
		}

	}

  	return true;
}
*/
int main(int argc, char **argv)
{
	ros::init(argc, argv, "pcl_server");
	ros::NodeHandle n;

	ros::ServiceServer service = n.advertiseService("batch_convert_pointcloud", batchConvertPC);
	ROS_INFO("Server Listening");
	ros::spin();

	return 0;
}
