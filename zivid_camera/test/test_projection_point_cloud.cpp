// Copyright 2026 Zivid AS
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the Zivid AS nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <tf2_ros/static_transform_broadcaster.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <zivid_camera/projection_point_cloud.hpp>

namespace
{
using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;

PointField field(const std::string & name, uint32_t offset, uint8_t datatype)
{
  PointField result;
  result.name = name;
  result.offset = offset;
  result.datatype = datatype;
  result.count = 1;
  return result;
}

template <typename T>
void writeValue(PointCloud2 & cloud, size_t offset, T value)
{
  std::array<uint8_t, sizeof(T)> bytes;
  std::memcpy(bytes.data(), &value, sizeof(value));
  const uint16_t one = 1;
  if (cloud.is_bigendian != (*reinterpret_cast<const uint8_t *>(&one) == 0)) {
    std::reverse(bytes.begin(), bytes.end());
  }
  std::copy(bytes.begin(), bytes.end(), cloud.data.begin() + offset);
}

PointCloud2 makeCloud(bool big_endian)
{
  PointCloud2 cloud;
  cloud.width = 2;
  cloud.height = 2;
  cloud.is_bigendian = big_endian;
  cloud.point_step = 24;
  cloud.row_step = 52;
  cloud.fields = {
    field("x", 0, PointField::FLOAT64), field("y", 8, PointField::FLOAT32),
    field("z", 12, PointField::FLOAT32), field("rgb", 16, PointField::FLOAT32)};
  cloud.data.resize(cloud.height * cloud.row_step);
  for (size_t row = 0; row < cloud.height; ++row) {
    for (size_t col = 0; col < cloud.width; ++col) {
      const auto offset = row * cloud.row_step + col * cloud.point_step;
      writeValue(cloud, offset, static_cast<double>(row + col));
      writeValue(cloud, offset + 8, 0.5F);
      writeValue(cloud, offset + 12, 1.0F);
      writeValue(cloud, offset + 16, uint32_t{0x00123456});
    }
  }
  return cloud;
}
}  // namespace

TEST(ProjectionPointCloud, LooksUpCloudTimestampAndDirection)
{
  tf2_ros::Buffer buffer{std::make_shared<rclcpp::Clock>()};
  buffer.setUsingDedicatedThread(true);
  geometry_msgs::msg::TransformStamped transform;
  transform.header.frame_id = "camera";
  transform.child_frame_id = "world";
  transform.transform.rotation.w = 1.0;
  transform.header.stamp.sec = 1;
  transform.transform.translation.x = 1.0;
  ASSERT_TRUE(buffer.setTransform(transform, "test"));
  transform.header.stamp.sec = 3;
  transform.transform.translation.x = 3.0;
  ASSERT_TRUE(buffer.setTransform(transform, "test"));

  std_msgs::msg::Header header;
  header.frame_id = "world";
  header.stamp.sec = 2;
  EXPECT_DOUBLE_EQ(zivid_camera::cameraFromCloud(buffer, "camera", header).getOrigin().x(), 2.0);
  header.stamp.sec = 0;
  EXPECT_DOUBLE_EQ(zivid_camera::cameraFromCloud(buffer, "camera", header).getOrigin().x(), 3.0);
  header.frame_id = "camera";
  EXPECT_DOUBLE_EQ(zivid_camera::cameraFromCloud(buffer, "camera", header).getOrigin().x(), 0.0);
  header.frame_id = "";
  EXPECT_THROW(zivid_camera::cameraFromCloud(buffer, "camera", header), std::runtime_error);
  header.frame_id = "world";
  header.stamp.sec = 4;
  EXPECT_THROW(zivid_camera::cameraFromCloud(buffer, "camera", header), tf2::TransformException);
}

TEST(ProjectionPointCloud, ReceivesStaticTransformsWhileNodeIsNotSpinning)
{
  auto node = std::make_shared<rclcpp::Node>("test_projection_tf_listener");
  tf2_ros::Buffer buffer{node->get_clock()};
  tf2_ros::TransformListener listener{buffer, node, true};
  tf2_ros::StaticTransformBroadcaster broadcaster{node};
  geometry_msgs::msg::TransformStamped transform;
  transform.header.frame_id = "camera";
  transform.child_frame_id = "world";
  transform.transform.rotation.w = 1.0;
  transform.transform.translation.z = 2.0;
  broadcaster.sendTransform(transform);

  std_msgs::msg::Header header;
  header.frame_id = "world";
  EXPECT_DOUBLE_EQ(zivid_camera::cameraFromCloud(buffer, "camera", header).getOrigin().z(), 2.0);
}

TEST(ProjectionPointCloud, TransformsPaddedCloudAndReadsBothByteOrders)
{
  for (const bool big_endian : {false, true}) {
    auto cloud = makeCloud(big_endian);
    writeValue(cloud, cloud.row_step + cloud.point_step, std::numeric_limits<double>::quiet_NaN());
    tf2::Transform transform;
    transform.setIdentity();
    transform.setRotation(tf2::Quaternion{0, 0, 1, 0});
    transform.setOrigin(tf2::Vector3{1, 2, 0.5});
    const auto result = zivid_camera::prepareProjectionPoints(cloud, transform);
    ASSERT_EQ(result.points.size(), 3U);
    EXPECT_FLOAT_EQ(result.points[0].x, 1000.0F);
    EXPECT_FLOAT_EQ(result.points[0].y, 1500.0F);
    EXPECT_FLOAT_EQ(result.points[0].z, 1500.0F);
    EXPECT_FLOAT_EQ(result.points[2].x, 0.0F);
    EXPECT_EQ(result.colors[0].r, 0x12);
    EXPECT_EQ(result.colors[0].g, 0x34);
    EXPECT_EQ(result.colors[0].b, 0x56);
  }
}

TEST(ProjectionPointCloud, ReadsRgbaIgnoringAlpha)
{
  auto cloud = makeCloud(false);
  cloud.fields.back() = field("rgba", 16, PointField::UINT32);
  writeValue(cloud, 16, uint32_t{0x78123456});
  const auto result = zivid_camera::prepareProjectionPoints(cloud, tf2::Transform::getIdentity());
  ASSERT_EQ(result.colors.size(), 4U);
  EXPECT_EQ(result.colors[0].r, 0x12);
  EXPECT_EQ(result.colors[0].g, 0x34);
  EXPECT_EQ(result.colors[0].b, 0x56);
  EXPECT_EQ(result.colors[0].a, 255);
}

TEST(ProjectionPointCloud, DefaultsToWhiteAndFiltersInvalidPoints)
{
  auto cloud = makeCloud(false);
  cloud.fields.pop_back();
  writeValue(cloud, 12, -1.0F);
  writeValue(cloud, cloud.point_step + 12, std::numeric_limits<float>::infinity());
  const auto result = zivid_camera::prepareProjectionPoints(cloud, tf2::Transform::getIdentity());
  ASSERT_EQ(result.points.size(), 2U);
  EXPECT_EQ(result.colors[0].r, 255);
  EXPECT_EQ(result.colors[0].g, 255);
  EXPECT_EQ(result.colors[0].b, 255);
}

TEST(ProjectionPointCloud, RejectsMalformedAndEmptyClouds)
{
  auto cloud = makeCloud(false);
  cloud.data.pop_back();
  EXPECT_THROW(
    zivid_camera::prepareProjectionPoints(cloud, tf2::Transform::getIdentity()),
    std::runtime_error);
  cloud = makeCloud(false);
  cloud.row_step = 1;
  EXPECT_THROW(
    zivid_camera::prepareProjectionPoints(cloud, tf2::Transform::getIdentity()),
    std::runtime_error);
  cloud = makeCloud(false);
  cloud.fields[0].offset = cloud.point_step;
  EXPECT_THROW(
    zivid_camera::prepareProjectionPoints(cloud, tf2::Transform::getIdentity()),
    std::runtime_error);
  cloud = makeCloud(false);
  cloud.fields[1].datatype = PointField::UINT8;
  EXPECT_THROW(
    zivid_camera::prepareProjectionPoints(cloud, tf2::Transform::getIdentity()),
    std::runtime_error);
  cloud = makeCloud(false);
  cloud.fields.erase(cloud.fields.begin());
  EXPECT_THROW(
    zivid_camera::prepareProjectionPoints(cloud, tf2::Transform::getIdentity()),
    std::runtime_error);
  cloud = makeCloud(false);
  for (size_t row = 0; row < cloud.height; ++row) {
    for (size_t col = 0; col < cloud.width; ++col) {
      writeValue(cloud, row * cloud.row_step + col * cloud.point_step + 12, -1.0F);
    }
  }
  EXPECT_THROW(
    zivid_camera::prepareProjectionPoints(cloud, tf2::Transform::getIdentity()),
    std::runtime_error);
}

TEST(ProjectionPointCloud, RendersColorsAndClipsInvalidPixels)
{
  const std::vector<Zivid::PointXY> pixels{
    {1.8F, 0.2F}, {1.0F, 0.0F}, {-0.1F, 0},
    {3, 0},       {0, 2},       {std::numeric_limits<float>::quiet_NaN(), 0}};
  const std::vector<Zivid::ColorBGRA> colors{{1, 2, 3, 255}, {4, 5, 6, 255}, {7, 8, 9, 255},
                                             {7, 8, 9, 255}, {7, 8, 9, 255}, {7, 8, 9, 255}};
  const auto image = zivid_camera::renderProjectionPoints(Zivid::Resolution{3, 2}, pixels, colors);
  EXPECT_EQ(image(0, 1).r, 6);
  EXPECT_EQ(image(0, 1).b, 4);
  EXPECT_EQ(image(0, 0).r, 0);
  EXPECT_EQ(image(1, 0).r, 0);
  EXPECT_THROW(
    zivid_camera::renderProjectionPoints(Zivid::Resolution{3, 2}, pixels, {}), std::runtime_error);
}
