package main

import (
	"fmt"
	"flag"
	"context"
	"strconv"
	"net/http"
	"io/ioutil"

	"github.com/labstack/echo"
	"google.golang.org/grpc"

	tf_framework "tensorflow/core/framework"
	pb "tensorflow_serving/apis"
)

var (
	port         int
	model_server string
	model_name   string
)

func init() {
	flag.IntVar(&port, "port", 1323, "concurrent processing ,default 1 .")
	flag.StringVar(&model_server, "model_server", "localhost:8500", "concurrent processing ,default 1 .")
	flag.StringVar(&model_name, "model_name", "default", "concurrent processing ,default 1 .")

	flag.Parse()
}

type Resp struct {
	Success bool
	Msg     string
	Result  [10]float32
}

func main() {
	e := echo.New()
	// Set up a connection to the model server.
	conn, err := grpc.Dial(model_server, grpc.WithInsecure())
	if err != nil {
		e.Logger.Fatalf("can not connect model_server: %v", err)
	}
	defer conn.Close()

	client := pb.NewPredictionServiceClient(conn)

	e.Static("/", "templates")
	e.Static("/static", "static")

	e.POST("/api/mnist", func(c echo.Context) error {
		req := c.Request()
		body, err := ioutil.ReadAll(req.Body)
		if err != nil {
			return err
		}
		result, err := Predict(client, body)
		if err != nil {
			e.Logger.Error(err.Error())
			return c.JSON(http.StatusOK, &Resp{
				Msg: err.Error(),
			})
		}

		return c.JSON(http.StatusOK, &Resp{
			Success: true,
			Result:  result,
		})
	})

	e.Logger.Fatal(e.Start(":" + strconv.Itoa(port)))
}

func Predict(c pb.PredictionServiceClient, imgBytes []byte) (result [10]float32, err error) {
	req := &pb.PredictRequest{
		ModelSpec: &pb.ModelSpec{Name: model_name},
		Inputs:    make(map[string]*tf_framework.TensorProto),
	}
	in := normalize(imgBytes)

	tp := &tf_framework.TensorProto{
		Dtype:    tf_framework.DataType_DT_FLOAT,
		FloatVal: in,
		TensorShape: &tf_framework.TensorShapeProto{
			Dim: []*tf_framework.TensorShapeProto_Dim{
				&tf_framework.TensorShapeProto_Dim{
					Size: int64(1),
					Name: "batch",
				},
				&tf_framework.TensorShapeProto_Dim{
					Size: int64(28),
					Name: "x",
				},
				&tf_framework.TensorShapeProto_Dim{
					Size: int64(28),
					Name: "y",
				},
				&tf_framework.TensorShapeProto_Dim{
					Size: int64(1),
					Name: "channel",
				},

			},
		},
	}
	req.Inputs["x"] = tp

	resp, err := c.Predict(context.Background(), req)
	if err != nil {
		return
	}
	output, ok := resp.Outputs["y"]
	if !ok {
		err = fmt.Errorf("can not find output data with label y")
		return
	}
	if len(output.FloatVal) != 10 {
		err = fmt.Errorf("wrong output dimension, it should be 10, now got %d", len(output.FloatVal))
		return
	}
	copy(result[:], output.FloatVal)
	return
}

func normalize(bytes []byte) (r []float32) {
	r = make([]float32, 0, len(bytes))
	for _, b := range bytes {
		r = append(r, float32(255-b)/255)
	}
	return
}
