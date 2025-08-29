import { Select, Form, Affix, FormInstance } from "antd";

const { Item } = Form;

type Props = {
  formInstance: FormInstance;
};

const IgnoredVideos = (formProps: Props) => {
  const { formInstance } = formProps;
  return (
    <div className="">
      <Item name="ignoredVideos" initialValue={[]}>
        <Select
          mode="multiple"
          placeholder="Ignored Videos"
          value={formInstance.getFieldValue("ignoredVideos")}
          style={{
            width: "100%",
            marginLeft: 2,
            borderWidth: 1,
            borderColor: "black",
          }}
        />
      </Item>
    </div>
  );
};

export default IgnoredVideos;
